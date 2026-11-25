#!/usr/bin/env python3
import csv
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

R_GRID = [1, 2, 5, 10, 20, 50, 100, 250, 500, 1000, 5000, 20000]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
REGRET_FORMULA = r"$\mathrm{Regret}(P) = \frac{C(P) - C(P_{\mathrm{oracle}})}{C(P_{\mathrm{oracle}})} \times 100\%$"
VALUE_FORMULA = r"$\mathrm{Value} = \frac{C(P_0) - C(P_{\mathrm{pred}})}{C(P_{\mathrm{oracle}})} \times 100\%$"


def pearson_r(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else 1.0


def mape_pct(pred, real):
    return float(np.mean(np.abs(pred - real) / np.maximum(1e-9, real)) * 100.0)


def draw_phase_panel(ax, grid, ms, ttl, cmap, norm):
    short = {0: "P0", 1: "P1", 2: "P2", 3: "P3"}
    ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap, norm=norm)
    for ri in range(len(R_GRID)):
        for mi in range(len(ms)):
            ax.text(mi, ri, short[int(grid[ri, mi])], ha="center", va="center", fontsize=6.5)
    ax.set_xticks(range(len(ms)))
    ax.set_xticklabels([str(m) for m in ms], rotation=45, fontsize=7.2)
    ax.set_yticks(range(len(R_GRID)))
    ax.set_yticklabels([str(r) for r in R_GRID], fontsize=7.8)
    ax.set_xlabel("Rows M (N=3072, K=768)")
    ax.set_ylabel("Model Runs R")
    ax.set_title(ttl, fontsize=9.6)


def draw_heat_panel(ax, fig, grid, ms, ttl, cmap_name, vmin, vmax, cbar_lbl):
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap_name, vmin=vmin, vmax=vmax)
    span = max(1e-6, vmax - vmin)
    for ri in range(len(R_GRID)):
        for mi in range(len(ms)):
            val = float(grid[ri, mi])
            txt = "0" if abs(val) < 0.05 else (f"{val:.1f}" if abs(val) < 10.0 else f"{val:.0f}")
            col = "white" if (val - vmin) > 0.58 * span else "#111111"
            ax.text(mi, ri, txt, ha="center", va="center", fontsize=6.1, color=col)
    ax.set_xticks(range(len(ms)))
    ax.set_xticklabels([str(m) for m in ms], rotation=45, fontsize=7.2)
    ax.set_yticks(range(len(R_GRID)))
    ax.set_yticklabels([str(r) for r in R_GRID], fontsize=7.8)
    ax.set_xlabel("Rows M (N=3072, K=768)")
    ax.set_ylabel("Model Runs R")
    ax.set_title(ttl, fontsize=9.2)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label=cbar_lbl)


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
            "pc": float(r["pred_compile_us"]),
            "rc": float(r["real_compile_us"]),
            "pe": float(r["pred_exec_us"]),
            "re": float(r["real_exec_us"]),
            "nomem_e": float(r["pred_nomem_exec_us"]),
            "nodisp_e": float(r["pred_nodisp_exec_us"]),
        }

    grid_p0 = np.zeros((len(R_GRID), len(ms)), dtype=int)
    grid_pred = np.zeros((len(R_GRID), len(ms)), dtype=int)
    grid_real = np.zeros((len(R_GRID), len(ms)), dtype=int)

    grid_reg_p0 = np.zeros((len(R_GRID), len(ms)), dtype=float)
    grid_reg_pred = np.zeros((len(R_GRID), len(ms)), dtype=float)
    grid_net_value = np.zeros((len(R_GRID), len(ms)), dtype=float)

    policies = {
        "Fixed: Always Split (P0)": [],
        "Fixed: Always Partial (P2)": [],
        "Fixed: Always Fuse (P3)": [],
        "Ablation: NoMemoryTraffic": [],
        "Ablation: NoDispatch": [],
        "Ablation: NoCompileCost": [],
        "Proposed: UniversalCostEngine": [],
    }

    matches_pred, matches_p0 = 0, 0
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

            calc_reg = lambda chosen: (real_cost[chosen] - oracle_c) / oracle_c * 100.0
            reg_p0 = calc_reg(0)
            reg_pred = calc_reg(full_p)

            grid_p0[ri, mi] = 0
            grid_real[ri, mi] = oracle_p
            grid_pred[ri, mi] = full_p

            grid_reg_p0[ri, mi] = reg_p0
            grid_reg_pred[ri, mi] = reg_pred
            grid_net_value[ri, mi] = reg_p0 - reg_pred

            if full_p == oracle_p:
                matches_pred += 1
            if oracle_p == 0:
                matches_p0 += 1

            policies["Fixed: Always Split (P0)"].append(reg_p0)
            policies["Fixed: Always Partial (P2)"].append(calc_reg(2))
            policies["Fixed: Always Fuse (P3)"].append(calc_reg(3))
            policies["Ablation: NoMemoryTraffic"].append(calc_reg(nomem_p))
            policies["Ablation: NoDispatch"].append(calc_reg(nodisp_p))
            policies["Ablation: NoCompileCost"].append(calc_reg(nocomp_p))
            policies["Proposed: UniversalCostEngine"].append(reg_pred)

    total_cells = len(ms) * len(R_GRID)
    agree_pred = matches_pred / total_cells * 100.0
    agree_p0 = matches_p0 / total_cells * 100.0

    mean_p0, p95_p0, max_p0 = float(np.mean(grid_reg_p0)), float(np.percentile(grid_reg_p0, 95)), float(np.max(grid_reg_p0))
    mean_pr, p95_pr, max_pr = float(np.mean(grid_reg_pred)), float(np.percentile(grid_reg_pred, 95)), float(np.max(grid_reg_pred))
    mean_val, max_val = float(np.mean(grid_net_value)), float(np.max(grid_net_value))

    print(f"\n==================== [{tag}] IDENTICAL-CONDITION VULKAN 1.3 VERIFICATION ====================")
    print(f"T_execute Correlation : Pearson r = {r_exec:.5f} (R2 = {r_exec**2:.5f}) | MAPE = {mape_e:.2f}%")
    print(f"T_compile Correlation : Pearson r = {r_comp:.5f} | MAPE = {mape_c:.2f}%")
    print(f"Phase Map Agreement   : Baseline P0 = {agree_p0:.1f}% -> UniversalCostEngine = {agree_pred:.1f}% ({matches_pred}/{total_cells})")
    print(f"Net Value vs P0       : Mean Savings = +{mean_val:.2f}% | Max Savings = +{max_val:.2f}% (Regret reduced {mean_p0/max(1e-4, mean_pr):.1f}x)")
    print("-" * 86)
    print(f"{'Policy / Ablation':<34} | {'Mean Regret (%)':>16} | {'P95 Regret (%)':>16} | {'Max Regret (%)':>14}")
    print("-" * 86)
    summary_stats = {}
    for name, regs in policies.items():
        arr = np.array(regs)
        summary_stats[name] = float(np.mean(arr))
        print(f"{name:<34} | {np.mean(arr):>15.3f}% | {np.percentile(arr, 95):>15.3f}% | {np.max(arr):>13.3f}%")

    # 2x3 Master Figure:
    # Top:    [Baseline All-P0 Map]      | [Predicted CostModel Map]        | [Measured Hardware Oracle Map]
    # Bottom: [All-P0 Regret vs Oracle]  | [CostModel Regret (Same Scale!)] | [Net Value Unlocked over P0 (+%)]
    fig, axs = plt.subplots(2, 3, figsize=(18.2, 9.6), dpi=180)
    cmap_p = mcolors.ListedColormap(["#6baed6", "#fdae6b", "#fd8d3c", "#74c476"])
    norm_p = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap_p.N)

    draw_phase_panel(axs[0, 0], grid_p0, ms, f"1. Baseline: Always Split (All P0)\nAgreement with Oracle = {agree_p0:.1f}%", cmap_p, norm_p)
    draw_phase_panel(axs[0, 1], grid_pred, ms, f"2. Proposed: UniversalCostEngine\nAgreement with Oracle = {agree_pred:.1f}%", cmap_p, norm_p)
    draw_phase_panel(axs[0, 2], grid_real, ms, "3. Measured Hardware Oracle (Vulkan 1.3)\nGround-Truth Optimal Partition P*", cmap_p, norm_p)

    shared_vmax = max(5.0, max_p0, max_pr)
    draw_heat_panel(
        axs[1, 0], fig, grid_reg_p0, ms,
        f"4. Baseline All-P0 Regret (%): Mean={mean_p0:.2f}%, P95={p95_p0:.1f}%, Max={max_p0:.1f}%\n" + REGRET_FORMULA,
        "YlOrRd", 0.0, shared_vmax, "Regret vs Oracle (%)",
    )
    draw_heat_panel(
        axs[1, 1], fig, grid_reg_pred, ms,
        f"5. Proposed CostModel Regret (%): Mean={mean_pr:.2f}%, P95={p95_pr:.2f}%, Max={max_pr:.1f}%\n" + REGRET_FORMULA,
        "YlOrRd", 0.0, shared_vmax, "Regret vs Oracle (%)",
    )
    draw_heat_panel(
        axs[1, 2], fig, grid_net_value, ms,
        f"6. Net Value Unlocked over All-P0 (+%): Mean=+{mean_val:.2f}%, Max=+{max_val:.1f}%\n" + VALUE_FORMULA,
        "YlGn", 0.0, shared_vmax, "Net Cost Savings vs P0 (%)",
    )

    fig.suptitle(
        f"[{tag}] Baseline All-P0 vs UniversalCostEngine vs Vulkan 1.3 Oracle "
        f"(Pearson r={r_exec:.4f}, MAPE={mape_e:.1f}%, Regret Cut {mean_p0/max(1e-4, mean_pr):.1f}x)",
        fontsize=12.2,
    )
    fig.tight_layout()
    out_p = out_dir / f"compare_phase_MR_{tag}.png"
    fig.savefig(out_p)
    plt.close(fig)
    print(f"Saved -> {out_p}")
    return tag, pred_e, real_e, r_exec, mape_e, summary_stats


def main():
    res_dir = Path(sys.argv[1]) if len(sys.argv[1:]) >= 1 else Path("results")
    vfiles = sorted(res_dir.glob("verify_*.csv"))
    if not vfiles:
        sys.exit(f"No verify_*.csv found in {res_dir}")

    fig_s, ax_s = plt.subplots(figsize=(6.8, 5.6), dpi=180)
    lo_all, hi_all = 1e9, 0.0
    all_stats = []
    for idx, vf in enumerate(vfiles):
        tag, pe, re_v, r_val, mp, st = analyze_file(vf, res_dir)
        all_stats.append((tag, st))
        c = COLORS[idx % len(COLORS)]
        ax_s.scatter(re_v, pe, color=c, s=32, alpha=0.82, label=f"{tag}: r={r_val:.4f}, MAPE={mp:.1f}%")
        lo_all, hi_all = min(lo_all, re_v.min(), pe.min()), max(hi_all, re_v.max(), pe.max())

    ax_s.plot([lo_all * 0.8, hi_all * 1.2], [lo_all * 0.8, hi_all * 1.2], "k--", lw=1.4, label="Ideal y = x")
    ax_s.set_xscale("log", base=10)
    ax_s.set_yscale("log", base=10)
    ax_s.set_xlabel("Measured Vulkan 1.3 Execution Time (us)")
    ax_s.set_ylabel("Predicted UniversalCostEngine Time (us)")
    ax_s.set_title("Blind Prediction vs Real Hardware Execution (P0..P3, M=16..4096)")
    ax_s.grid(True, which="both", ls=":", alpha=0.6)
    ax_s.legend(fontsize=8.5)
    fig_s.tight_layout()
    out_c = res_dir / "corr_pred_vs_meas.png"
    fig_s.savefig(out_c)
    plt.close(fig_s)
    print(f"\nSaved -> {out_c}")

    if all_stats:
        pol_names = list(all_stats[0][1].keys())
        x = np.arange(len(pol_names))
        w = 0.36 / max(1, len(all_stats) / 2)
        fig_b, ax_b = plt.subplots(figsize=(10.5, 5.2), dpi=180)
        for idx, (tag, st) in enumerate(all_stats):
            vals = [max(0.01, st[k]) for k in pol_names]
            offset = (idx - (len(all_stats) - 1) / 2.0) * w
            bars = ax_b.bar(x + offset, vals, w, color=COLORS[idx % len(COLORS)], label=tag)
            for b, raw_v in zip(bars, [st[k] for k in pol_names]):
                ax_b.text(
                    b.get_x() + b.get_width() / 2.0,
                    b.get_height() * 1.08,
                    f"{raw_v:.2f}%",
                    ha="center",
                    va="bottom",
                    fontsize=7.0,
                    rotation=25,
                )
        ax_b.set_yscale("log", base=10)
        ax_b.set_xticks(x)
        ax_b.set_xticklabels(pol_names, rotation=20, ha="right", fontsize=8.5)
        ax_b.set_ylabel("Mean Regret (%, log10 scale — lower is better)")
        ax_b.set_title("Strategy & Ablation Comparison: " + REGRET_FORMULA, fontsize=10.5)
        ax_b.grid(axis="y", which="both", ls=":", alpha=0.6)
        ax_b.legend(fontsize=8.8)
        fig_b.tight_layout()
        out_b = res_dir / "regret_policies_bar.png"
        fig_b.savefig(out_b)
        plt.close(fig_b)
        print(f"Saved -> {out_b}")


if __name__ == "__main__":
    main()