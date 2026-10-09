#!/usr/bin/env python3
"""第 2 步:从快照复现团队的两份预测,并逐点对照 W&B 上的原值。

A. 预注册(8/19,终点 2.039):照 refs/plot_scaling_ladder@d23e6e9.py 的做法。
   GB200 阶梯 d768–d2048,每 5% 单独拟合 log(L − 1.5) ~ log C;d2048 80% 以后用 60–80% 线性外推 + 0.005。
B. 10/3 重拟合(终点 2.100 / 2.113):照该 run config 里写的做法。
   H100 阶梯 d512–d1536、新旧两种配比、30%–100% 共 15 个进度点;loss 空间最小二乘;
   一个非负下界 F 在所有进度点和两种配比间共用;振幅 A 和指数 α(界于 [0, 5])按 (配比, 进度) 各自拟合。

用法:
  python reproduce.py ../data/snapshot_2026-10-09 ../results
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize_scalar

KEY = "eval_dropless/paloma/macro_loss"
HERO_STEPS = 390_251

# ---------- A. 预注册 ----------
PRE_RUNS = {"d768": "rav-ladder-d768-v2", "d1024": "rav-ladder-d1024", "d1536": "rav-ladder-d1536", "d2048": "rav-ladder-d2048-v3"}
# 照抄团队脚本的 FLOPS 常数(不含 lm_head)
PRE_FLOPS = {
    "d768": (2.320210052775936e15, 11420),
    "d1024": (1.129954355970048e16, 15276),
    "d1536": (9.130921800656486e16, 15128),
    "d2048": (3.857022193930076e17, 20072),
    "d6144": (6.690294608796058e18, 390251),
}
PRE_E, D2048_CORRECTION = 1.5, 0.005

# ---------- B. 10/3 重拟合 ----------
REFIT_FRACTIONS = [round(0.30 + 0.05 * i, 2) for i in range(15)]
REFIT_HERO_C = 2.6960230106658617e24  # config: source_compute["d6144-old"](含 lm_head 的口径)
REFIT_TEAM_RUNS = {"new": "hero-dropless-refit-new-20261003-aa4pdwep", "old": "hero-dropless-refit-old-20261003-phq0qbmf"}


def load(snap):
    long = pd.read_csv(os.path.join(snap, "tables", "wandb_long.csv"))
    runs = pd.read_csv(os.path.join(snap, "tables", "runs.csv")).set_index("run")
    return long, runs


def series(long, run, key=KEY):
    ev = long[(long.run == run) & (long.key == key)].sort_values("step")
    return list(zip(ev.step.astype(int), ev.value.astype(float)))


# ======================= A =======================
def prereg_grid(long, run, stop):
    """团队脚本的吸附规则:round(round(step/stop*20)/20, 2),同一格后者覆盖前者。"""
    grid = {}
    for step, v in series(long, run):
        grid[round(round(step / stop * 20) / 20, 2)] = v
    return grid


def reproduce_prereg(long, runs):
    grids = {r: prereg_grid(long, name, int(runs.loc[name, "stop_after_steps"])) for r, name in PRE_RUNS.items()}
    g2048 = grids["d2048"]
    window = [0.60, 0.65, 0.70, 0.75, 0.80]
    slope, intercept = np.polyfit(np.array(window), np.array([g2048[w] for w in window]), 1)
    d2048_max = max(g2048)

    def loss_at(r, g):
        if r == "d2048" and g > d2048_max:
            return slope * g + intercept + D2048_CORRECTION
        return grids[r][round(g, 2)]

    rows = []
    for g in [round(x, 2) for x in np.arange(0.05, 1.0001, 0.05)]:
        cs = np.array([PRE_FLOPS[r][0] * PRE_FLOPS[r][1] * g for r in PRE_RUNS])
        ls = np.array([loss_at(r, g) for r in PRE_RUNS])
        s, i = np.polyfit(np.log(cs), np.log(ls - PRE_E), 1)
        a, alpha = np.exp(i), -s
        c_hero = PRE_FLOPS["d6144"][0] * PRE_FLOPS["d6144"][1] * g
        rows.append({"fraction": g, "A": a, "alpha": alpha, "ours": PRE_E + a * c_hero ** (-alpha),
                     **{f"L_{r}": loss_at(r, g) for r in PRE_RUNS}})
    out = pd.DataFrame(rows)
    team = pd.DataFrame(series(long, "hero-forecast-seqlen4096-fc"), columns=["step", "team"])
    team["fraction"] = (team.step / HERO_STEPS).round(2)
    out = out.merge(team[["fraction", "team"]], on="fraction", how="left")
    out["diff"] = out.ours - out.team
    return out, {"d2048_slope": slope, "d2048_intercept": intercept, "d2048_pinned_final": slope + intercept + D2048_CORRECTION}


# ======================= B =======================
def refit_grid(long, run, stop):
    """10/3 重拟合 config 的吸附规则:(zero-indexed step + 1)/stop 取最近的 5% 格,同一格后者覆盖前者。"""
    grid = {}
    for step, v in series(long, run):
        grid[round(round((step + 1) / stop * 20) / 20, 2)] = v
    return grid


def refit_data(long, runs, snap):
    meta = json.load(open(os.path.join(snap, "raw", "wandb", "forecast", REFIT_TEAM_RUNS["new"], "meta.json")))
    cfg = {k: (v.get("value") if isinstance(v, dict) and "value" in v else v) for k, v in meta["config"].items()}
    src_runs, src_c = cfg["source_runs"], cfg["source_compute"]
    groups = {}  # (mix, fraction) -> list of (width, C, L)
    for tag, run in src_runs.items():
        width, mix = tag.split("-")
        if width == "d6144":
            continue
        grid = refit_grid(long, run, int(runs.loc[run, "stop_after_steps"]))
        for p in REFIT_FRACTIONS:
            if p in grid:
                groups.setdefault((mix, p), []).append((width, src_c[tag] * p, grid[p]))
    return groups, cfg


def fit_group(F, pts):
    """给定下界 F,在 loss 空间拟合 L = F + A·C^(−α),A>0、α∈[0,5]。返回 (A, α, SSE)。"""
    c = np.array([p[1] for p in pts])
    y = np.array([p[2] for p in pts])
    cn = c / c.min()  # 归一化,数值更稳
    z = np.maximum(y - F, 1e-6)
    s, i = np.polyfit(np.log(cn), np.log(z), 1)  # 初值:log-log 线性拟合
    x0 = [i, float(np.clip(-s, 1e-4, 4.999))]

    def resid(x):
        return F + np.exp(x[0]) * cn ** (-x[1]) - y

    sol = least_squares(resid, x0, bounds=([-50, 0.0], [50, 5.0]), xtol=1e-14, ftol=1e-14, gtol=1e-14)
    a_norm, alpha = np.exp(sol.x[0]), sol.x[1]
    A = a_norm * c.min() ** alpha  # 换回原始 C 的尺度
    return A, alpha, float(np.sum(sol.fun ** 2))


def total_sse(F, groups):
    return sum(fit_group(F, pts)[2] for pts in groups.values())


def reproduce_refit(long, runs, snap):
    groups, cfg = refit_data(long, runs, snap)
    lmin = min(p[2] for pts in groups.values() for p in pts)
    res = minimize_scalar(lambda F: total_sse(F, groups), bounds=(0.0, lmin - 1e-4), method="bounded",
                          options={"xatol": 1e-7})
    F = res.x
    sse_min = res.fun
    # 团队写的敏感性:SSE 不超过最小值 1.1 倍的下界范围
    grid = np.linspace(max(0.0, F - 0.8), min(lmin - 1e-4, F + 0.8), 801)
    sse = np.array([total_sse(f, groups) for f in grid])
    ok = grid[sse <= 1.1 * sse_min]
    rows = []
    for (mix, p), pts in sorted(groups.items()):
        A, alpha, _ = fit_group(F, pts)
        rows.append({"mix": mix, "fraction": p, "n_widths": len(pts), "A": A, "alpha": alpha,
                     "ours": F + A * (REFIT_HERO_C * p) ** (-alpha)})
    out = pd.DataFrame(rows)
    team_rows = []
    for mix, run in REFIT_TEAM_RUNS.items():
        for step, v in series(long, run):
            team_rows.append({"mix": mix, "fraction": round(step / HERO_STEPS, 2), "team": v})
    out = out.merge(pd.DataFrame(team_rows), on=["mix", "fraction"], how="left")
    out["diff"] = out.ours - out.team
    n_pts = sum(len(v) for v in groups.values())
    info = {"floor": F, "sse": sse_min, "rmse": float(np.sqrt(sse_min / n_pts)), "n_points": n_pts,
            "floor_range_sse_le_1.1x": [float(ok.min()), float(ok.max())] if len(ok) else None,
            "team_floor": json.load(open(os.path.join(snap, "raw", "wandb", "forecast", REFIT_TEAM_RUNS["new"], "meta.json")))
            ["summaryMetrics"].get("fit/paloma/macro_loss/floor"),
            "team_fit_rmse": json.load(open(os.path.join(snap, "raw", "wandb", "forecast", REFIT_TEAM_RUNS["new"], "meta.json")))
            ["summaryMetrics"].get("fit/paloma/macro_loss/fit_rmse")}
    return out, info, groups


def hero_actual(long):
    ev = long[(long.category == "hero") & (long.key == KEY)].drop_duplicates("step").sort_values("step")
    ev = ev[ev.step > 0]
    return pd.DataFrame({"fraction": (ev.step + 1) / HERO_STEPS, "loss": ev.value})


def plot(pre, ref, actual, path):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.plot(actual.fraction * 100, actual.loss, "-", color="black", lw=1.8, label="hero 实测(dropless Paloma macro)")
    ax.plot(pre.fraction * 100, pre.team, "o", color="crimson", ms=7, mfc="none", label="团队预注册 8/19(W&B)")
    ax.plot(pre.fraction * 100, pre.ours, "-", color="crimson", lw=1.2, label="我们复现的预注册")
    for mix, color in (("new", "#1f77b4"), ("old", "#2ca02c")):
        d = ref[ref.mix == mix]
        ax.plot(d.fraction * 100, d.team, "s", color=color, ms=6, mfc="none", label=f"团队重拟合 10/3 · {mix} 配比(W&B)")
        ax.plot(d.fraction * 100, d.ours, "-", color=color, lw=1.2, label=f"我们复现的重拟合 · {mix}")
    ax.set_xlabel("训练进度(%)")
    ax.set_ylabel("dropless Paloma macro loss")
    ax.set_title("第 2 步:复现团队的两份预测")
    ax.set_xlim(0, 102)
    ax.set_ylim(2.0, 2.45)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main(snap, outdir):
    os.makedirs(outdir, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    long, runs = load(snap)

    pre, pre_info = reproduce_prereg(long, runs)
    pre.to_csv(os.path.join(outdir, "step2_prereg.csv"), index=False)
    print("== A. 预注册(8/19)==")
    print(pre[["fraction", "A", "alpha", "ours", "team", "diff"]].round(5).to_string(index=False))
    print(f"终点:我们 {pre.ours.iloc[-1]:.4f},团队 {pre.team.iloc[-1]:.4f};逐点最大偏差 {pre['diff'].abs().max():.5f}")
    print("d2048 外推:", {k: round(v, 5) for k, v in pre_info.items()})

    ref, ref_info, groups = reproduce_refit(long, runs, snap)
    ref.to_csv(os.path.join(outdir, "step2_refit.csv"), index=False)
    print("\n== B. 10/3 重拟合 ==")
    print(f"下界 F:我们 {ref_info['floor']:.4f},团队 {ref_info['team_floor']:.4f}")
    print(f"拟合 rmse:我们 {ref_info['rmse']:.6f}(n={ref_info['n_points']}),团队 {ref_info['team_fit_rmse']:.6f}")
    print(f"SSE ≤ 1.1×最小值 的下界范围:{ref_info['floor_range_sse_le_1.1x']}")
    print(ref[["mix", "fraction", "n_widths", "alpha", "ours", "team", "diff"]].round(5).to_string(index=False))
    for mix in ("new", "old"):
        d = ref[ref.mix == mix]
        print(f"终点 {mix}:我们 {d.ours.iloc[-1]:.4f},团队 {d.team.iloc[-1]:.4f};逐点最大偏差 {d['diff'].abs().max():.5f}")

    actual = hero_actual(long)
    actual.to_csv(os.path.join(outdir, "hero_actual.csv"), index=False)
    plot(pre, ref, actual, os.path.join(outdir, "step2_reproduction.png"))
    json.dump({"prereg": pre_info, "refit": ref_info}, open(os.path.join(outdir, "step2_summary.json"), "w"), indent=1)
    print(f"\n结果写在 {outdir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
