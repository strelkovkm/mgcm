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
VALUE_FORMULA = r"$\mathrm{Value\ vs\ P_0} = \frac{C(P_0) - C(P_{\mathrm{pred}})}{C(P_{\mathrm{oracle}})} \times 100\%$"


def pearson_r(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else 1.0


def mape_pct(pred, real):
    return float(np.mean(np.abs(pred - real) / np.maximum(1e-9, real)) * 100.0)


def draw_phase_panel(ax, grid, ms, ttl, cmap, norm):
    short = {0: "P0", 1: "P1", 2: "P2", 3: "P3"}
    ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap, norm=norm)
    for ri in range(len(R_GRID)):
        for mi in range(len(ms)):
            ax.text(mi, ri, short[int(grid[ri, mi])], ha="center", va="center", fontsize=6.0)
    ax.set_xticks(range(len(ms)))
    ax.set_xticklabels([str(m) for m in ms], rotation=45, fontsize=6.8)
    ax.set_yticks(range(len(R_GRID)))
    ax.set_yticklabels([str(r) for r in R_GRID], fontsize=7.2)
    ax.set_xlabel("Rows M")
    ax.set_ylabel("Model Runs R")
    ax.set_title(ttl, fontsize=9.0)


def draw_heat_panel(ax, fig, grid, ms, ttl, cmap_name, norm, cbar_lbl, is_log=False):
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap_name, norm=norm)
    for ri in range(len(R_GRID)):
        for mi in range(len(ms)):
            val = float(grid[ri, mi])
            if abs(val) < 0.05:
                txt = "0"
            elif abs(val) < 10.0:
                txt = f"{val:.1f}"
            elif abs(val) < 1000.0:
                txt = f"{val:.0f}"
            else:
                txt = f"{val / 1e3:.0f}k"
            if is_log:
                col = "white" if val > 0.35 * norm.vmax ** 0.55 else "#111111"
            else:
                span = max(1e-6, norm.vmax - norm.vmin)
                col = "white" if (val - norm.vmin) > 0.58 * span else "#111111"
            ax.text(mi, ri, txt, ha="center", va="center", fontsize=5.6, color=col)
    ax.set_xticks(range(len(ms)))
    ax.set_xticklabels([str(m) for m in ms], rotation=45, fontsize=6.8)
    ax.set_yticks(range(len(R_GRID)))
    ax.set_yticklabels([str(r) for r in R_GRID], fontsize=7.2)
    ax.set_xlabel("Rows M")
    ax.set_ylabel("Model Runs R")
    ax.set_title(ttl, fontsize=8.4)
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
            "pc": float(r["pred_compile_us"]), "rc": float(r["real_compile_us"]),
            "pe": float(r["pred_exec_us"]), "re": float(r["real_exec_us"]),
            "nomem_e": float(r["pred_nomem_exec_us"]), "nodisp_e": float(r["pred_nodisp_exec_us"]),
        }

    grid_p0 = np.zeros((len(R_GRID), len(ms)), dtype=int)
    grid_p3 = np.full((len(R_GRID), len(ms)), 3, dtype=int)
    grid_pred = np.zeros((len(R_GRID), len(ms)), dtype=int)
    grid_real = np.zeros((len(R_GRID), len(ms)), dtype=int)

    grid_reg_p0 = np.zeros((len(R_GRID), len(ms)), dtype=float)
    grid_reg_p3 = np.zeros((len(R_GRID), len(ms)), dtype=float)
    grid_reg_pred = np.zeros((len(R_GRID), len(ms)), dtype=float)
    grid_net_value = np.zeros((len(R_GRID), len(ms)), dtype=float)

    policies = {
        "Fixed: Always Split (P0)": [], "Fixed: Always Partial (P2)": [], "Fixed: Always Fuse (P3)": [],
        "Ablation: NoMemoryTraffic": [], "Ablation: NoDispatch": [], "Ablation: NoCompileCost": [],
        "Proposed: UniversalCostEngine": [],
    }

    matches_pred, matches_p0, matches_p3 = 0, 0, 0
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
            reg_p0, reg_p3, reg_pred = calc_reg(0), calc_reg(3), calc_reg(full_p)

            grid_real[ri, mi], grid_pred[ri, mi] = oracle_p, full_p
            grid_reg_p0[ri, mi], grid_reg_p3[ri, mi], grid_reg_pred[ri, mi] = reg_p0, reg_p3, reg_pred
            grid_net_value[ri, mi] = reg_p0 - reg_pred

            if full_p == oracle_p: matches_pred += 1
            if oracle_p == 0: matches_p0 += 1
            if oracle_p == 3: matches_p3 += 1

            policies["Fixed: Always Split (P0)"].append(reg_p0)
            policies["Fixed: Always Partial (P2)"].append(calc_reg(2))
            policies["Fixed: Always Fuse (P3)"].append(reg_p3)
            policies["Ablation: NoMemoryTraffic"].append(calc_reg(nomem_p))
            policies["Ablation: NoDispatch"].append(calc_reg(nodisp_p))
            policies["Ablation: NoCompileCost"].append(calc_reg(nocomp_p))
            policies["Proposed: UniversalCostEngine"].append(reg_pred)

    tot = len(ms) * len(R_GRID)
    agree_pred, agree_p0, agree_p3 = matches_pred / tot * 100.0, matches_p0 / tot * 100.0, matches_p3 / tot * 100.0
    mean_p0, max_p0 = float(np.mean(grid_reg_p0)), float(np.max(grid_reg_p0))
    mean_p3, max_p3 = float(np.mean(grid_reg_p3)), float(np.max(grid_reg_p3))
    mean_pr, max_pr = float(np.mean(grid_reg_pred)), float(np.max(grid_reg_pred))
    mean_val, max_val = float(np.mean(grid_net_value)), float(np.max(grid_net_value))

    print(f"\n==================== [{tag}] IDENTICAL-CONDITION VULKAN 1.3 VERIFICATION ====================")
    print(f"T_execute Correlation : Pearson r = {r_exec:.5f} (R2 = {r_exec**2:.5f}) | MAPE = {mape_e:.2f}%")
    print(f"T_compile Correlation : Pearson r = {r_comp:.5f} | MAPE = {mape_c:.2f}%")
    print(f"Phase Map Agreement   : All-P0 = {agree_p0:.1f}% | All-P3 = {agree_p3:.1f}% -> CostModel = {agree_pred:.1f}%")
    print(f"Regret Reduction      : {mean_p0/max(1e-4, mean_pr):.1f}x vs All-P0 | {mean_p3/max(1e-4, mean_pr):.1f}x vs All-P3")
    summary_stats = {}
    for name, regs in policies.items():
        arr = np.array(regs)
        summary_stats[name] = float(np.mean(arr))
        print(f"  {name:<32} Mean={np.mean(arr):>9.3f}% | P95={np.percentile(arr, 95):>9.3f}% | Max={np.max(arr):>9.3f}%")

    # 2x4 Master Grid: P0 | P3 | CostModel | Oracle (top), and their Regret/Value maps (bottom)
    fig, axs = plt.subplots(2, 4, figsize=(22.5, 9.2), dpi=170)
    cmap_p = mcolors.ListedColormap(["#6baed6", "#fdae6b", "#fd8d3c", "#74c476"])
    norm_p = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap_p.N)

    draw_phase_panel(axs[0, 0], grid_p0, ms, f"1. Fixed: Always Split (All P0)\nAgreement = {agree_p0:.1f}%", cmap_p, norm_p)
    draw_phase_panel(axs[0, 1], grid_p3, ms, f"2. Fixed: Always Fuse (All P3)\nAgreement = {agree_p3:.1f}%", cmap_p, norm_p)
    draw_phase_panel(axs[0, 2], grid_pred, ms, f"3. Proposed: UniversalCostEngine\nAgreement = {agree_pred:.1f}%", cmap_p, norm_p)
    draw_phase_panel(axs[0, 3], grid_real, ms, "4. Measured Hardware Oracle\nGround-Truth Optimal P*", cmap_p, norm_p)

    shared_vmax = max(5.0, max_p0, max_pr)
    norm_lin = mcolors.Normalize(vmin=0.0, vmax=shared_vmax)
    norm_p3_log = mcolors.SymLogNorm(linthresh=1.0, vmin=0.0, vmax=max(10.0, max_p3))

    draw_heat_panel(axs[1, 0], fig, grid_reg_p0, ms,
                    f"5. All-P0 Regret: Mean={mean_p0:.2f}%, Max={max_p0:.1f}%\n" + REGRET_FORMULA,
                    "YlOrRd", norm_lin, "Regret vs Oracle (%)")
    draw_heat_panel(axs[1, 1], fig, grid_reg_p3, ms,
                    f"6. All-P3 Regret (Log): Mean={mean_p3:.0f}%, Max={max_p3:.0f}%\n" + REGRET_FORMULA,
                    "YlOrRd", norm_p3_log, "Regret (%, log scale)", is_log=True)
    draw_heat_panel(axs[1, 2], fig, grid_reg_pred, ms,
                    f"7. CostModel Regret: Mean={mean_pr:.2f}%, Max={max_pr:.1f}%\n" + REGRET_FORMULA,
                    "YlOrRd", norm_lin, "Regret vs Oracle (%)")
    draw_heat_panel(axs[1, 3], fig, grid_net_value, ms,
                    f"8. Net Value vs All-P0: Mean=+{mean_val:.2f}%, Max=+{max_val:.1f}%\n" + VALUE_FORMULA,
                    "YlGn", norm_lin, "Savings vs P0 (%)")

    fig.suptitle(
        f"[{tag}] All-P0 vs All-P3 vs UniversalCostEngine vs Vulkan 1.3 Oracle "
        f"(Pearson r={r_exec:.4f}, MAPE={mape_e:.1f}%)", fontsize=12.5)
    fig.tight_layout()
    out_p = out_dir / f"compare_phase_MR_{tag}.png"
    fig.savefig(out_p); plt.close(fig)
    print(f"Saved -> {out_p}")
    return tag, pred_e, real_e, r_exec, mape_e, summary_stats


def main():
    res_dir = Path(sys.argv[1]) if len(sys.argv[1:]) >= 1 else Path("results")
    vfiles = sorted(res_dir.glob("verify_*.csv"))
    if not vfiles: sys.exit(f"No verify_*.csv found in {res_dir}")
    fig_s, ax_s = plt.subplots(figsize=(6.8, 5.6), dpi=180)
    lo_all, hi_all, all_stats = 1e9, 0.0, []
    for idx, vf in enumerate(vfiles):
        tag, pe, re_v, r_val, mp, st = analyze_file(vf, res_dir)
        all_stats.append((tag, st))
        c = COLORS[idx % len(COLORS)]
        ax_s.scatter(re_v, pe, color=c, s=32, alpha=0.82, label=f"{tag}: r={r_val:.4f}, MAPE={mp:.1f}%")
        lo_all, hi_all = min(lo_all, re_v.min(), pe.min()), max(hi_all, re_v.max(), pe.max())
    ax_s.plot([lo_all * 0.8, hi_all * 1.2], [lo_all * 0.8, hi_all * 1.2], "k--", lw=1.4, label="Ideal y = x")
    ax_s.set_xscale("log", base=10); ax_s.set_yscale("log", base=10)
    ax_s.set_xlabel("Measured Vulkan 1.3 Execution Time (us)"); ax_s.set_ylabel("Predicted Time (us)")
    ax_s.set_title("Blind Prediction vs Real Hardware Execution (P0..P3, M=16..4096)")
    ax_s.grid(True, which="both", ls=":", alpha=0.6); ax_s.legend(fontsize=8.5)
    fig_s.tight_layout(); fig_s.savefig(res_dir / "corr_pred_vs_meas.png"); plt.close(fig_s)

    if all_stats:
        pol_names = list(all_stats[0][1].keys())
        x, w = np.arange(len(pol_names)), 0.36 / max(1, len(all_stats) / 2)
        fig_b, ax_b = plt.subplots(figsize=(10.5, 5.2), dpi=180)
        for idx, (tag, st) in enumerate(all_stats):
            vals = [max(0.01, st[k]) for k in pol_names]
            bars = ax_b.bar(x + (idx - (len(all_stats) - 1) / 2.0) * w, vals, w, color=COLORS[idx % len(COLORS)], label=tag)
            for b, rv in zip(bars, [st[k] for k in pol_names]):
                ax_b.text(b.get_x() + b.get_width() / 2.0, b.get_height() * 1.08, f"{rv:.2f}%", ha="center", va="bottom", fontsize=6.8, rotation=25)
        ax_b.set_yscale("log", base=10); ax_b.set_xticks(x); ax_b.set_xticklabels(pol_names, rotation=20, ha="right", fontsize=8.5)
        ax_b.set_ylabel("Mean Regret (%, log10 scale)"); ax_b.set_title("Strategy & Ablation Comparison: " + REGRET_FORMULA, fontsize=10.5)
        ax_b.grid(axis="y", which="both", ls=":", alpha=0.6); ax_b.legend(fontsize=8.8)
        fig_b.tight_layout(); fig_b.savefig(res_dir / "regret_policies_bar.png"); plt.close(fig_b)


if __name__ == "__main__":
    main()