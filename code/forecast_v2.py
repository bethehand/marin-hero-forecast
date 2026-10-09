#!/usr/bin/env python3
"""v2(2026-10-09):按独立审查和回测证据修正 v1 的方法。v1 的代码和结果保留不动,数据也和 v1 相同。

v2 相对 v1 的改动(依据见 preregistration_v2.md 第 6 节):
  1. 漂移:κ ~ U[0, κ_max]。κ_max 由回测决定:15 子集伪 hero 误差的均方根不超过最小值 1.1 倍的最大 κ。v1 是 U[0, 1]。
  2. d2048 在 80% 以后的外推:改用已完成 GB200 rung(d768、d1024、d1536)在 85/90/95/100% 处
     相对各自 60–80% 线性外推的实测偏差,三套剖面等权。v1 是在 85–100% 一律加 U[0.003, 0.006]。
  3. 方法误差 ε:回测用的估计量和预测时相同(κ 取先验均值;v1 的回测相当于 κ = 0);
     新增两个外推 52 倍的伪 hero;15 子集的误差池把 67B(外推 166 倍,只有 16 子集数据)也算进去;
     ε 服从 t 分布,自由度 = 案例数 − 1,反映案例少带来的不确定。
  4. 外推距离偏差:回测误差随 log10(外推距离) 上升(方法越远越偏悲观),但在案例层面不显著。按这条趋势在 hero
     外推距离处的值 b(d) 设偏差 β ~ U[0, b(d)],从预测里减掉:一端是没有偏差,一端是完整的回测趋势。
  5. PTB:长期影响 r2 ~ U[−0.06, 0] 只从第二次曝光(#9874:第 362,280–362,578 步)起计入;v1 是 U[−0.06, 0],
     按剩余进度线性计入。当前水平 D_now 的标准误按残差的一阶自相关放宽。新增对官方 16 子集这个组合目标本身的回测和校准。
  6. 留一校准:每一折里 κ_max、σ_ε 和距离趋势都只用其余案例来定;67B 的评测噪声取 hero 16 子集的量级。
  7. 训练被缩短时,PTB 差值的变化按在新终点完成衰减计(v1 按剩余进度的比例缩小);第二次曝光之前结束的不计 r2。
  8. 同一步有多个 run 的评测时取血缘最新的 run;轨迹表的 step 满足 (step + 1)/390,251 = 进度,
     并在第二次曝光前后各加一个点,使插值时 r2 不会提前计入。

用法:
  python forecast_v2.py ../data/snapshot_2026-10-09T0440Z ../results/v2
写好预注册文档以后:
  python forecast_v2.py --hash-only ../results/v2 ../preregistration_v2.md
"""

import hashlib
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import forecast_step3 as F
from fetch_snapshot import RUNS as FETCH_RUNS
from reproduce import load

HERE = os.path.dirname(os.path.abspath(__file__))
KAPPA_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
KAPPA_TOL = 1.10
PTB_V1 = (-0.06, 0.0)  # v1 的长期影响项(按剩余进度线性计入),只在 v1→v2 对照里用
PTB_R2 = (-0.06, 0.0)  # 第二次曝光可能留下的长期影响
EXPO2 = (362_280, 362_578)  # #9874:第二次读到 Paloma PTB 文本的步数
P_EXPO2 = (EXPO2[0] + 1) / F.HERO_STEPS
ACUTE_STEPS = 12_000  # 打分规则里的急性期长度(曝光结束后)
N_MC = 60_000
SEED = 20261011
TEAM_EN = {"pre-registered (Aug 19)": 2.039, "swap-adjusted (Sep 14)": 2.02685, "refit, new mix (Oct 3)": 2.1001}
DIST_BINS = [(0, 30, "6.5–8x"), (30, 100, "52x"), (100, 1e9, "166x (67B)")]
CODE_FILES = ["fetch_snapshot.py", "check_snapshot.py", "reproduce.py", "diagnose_floor.py", "explore_step3.py",
              "forecast_step3.py", "robustness_step3.py", "perturbations_step4.py", "freeze_step5.py", "forecast_v2.py"]


# ---------------- 数据 ----------------
def dedupe_latest(long):
    """同一步、同一指标在多个 hero run 里都有时,取血缘里最新的 run(回滚后重跑的那一段)。"""
    order = {r: i for i, r in enumerate(FETCH_RUNS["hero"])}
    h = long[long.category == "hero"].copy()
    h["_ord"] = h.run.map(order)
    h = h.sort_values(["step", "key", "_ord"]).drop_duplicates(["step", "key"], keep="last").drop(columns="_ord")
    return pd.concat([long[long.category != "hero"], h], ignore_index=True)


def d2048_profiles(lads, target):
    G = lads["GB200"]["grids"][target]
    win = [0.60, 0.65, 0.70, 0.75, 0.80]
    out = {}
    for w in ("d768", "d1024", "d1536"):
        g = G[w]
        k, b = np.polyfit(win, [g[x] for x in win], 1)
        out[w] = {p: float(g[p] - (k * p + b)) for p in F.FRACS if p > 0.80}
    return out


def extend_d2048_profile(grid, prof):
    g = dict(grid)
    win = [0.60, 0.65, 0.70, 0.75, 0.80]
    k, b = np.polyfit(win, [g[x] for x in win], 1)
    for p in F.FRACS:
        if p > 0.80:
            g[p] = k * p + b + prof[p]
    return g


def max_rung_c(lad, widths, conv="C_nolm"):
    return max(lad["rungs"][w][conv] for w in widths)


def d_grids(long, runs, lads):
    """每个 rung 的 D = PTB − 15 子集 macro,按 5% 进度对齐。"""
    out = {}
    for name, lad in lads.items():
        out[name] = {}
        for w, v in lad["rungs"].items():
            g15 = lad["grids"]["macro15"][w]
            gp = F.snap(F.key_steps(long, v["run"], F.PTB_KEY), int(runs.loc[v["run"], "stop_after_steps"]), lad["rule"])
            out[name][w] = {p: gp[p] - g15[p] for p in g15 if p in gp}
    return out


def d_now_fit(D, p_now):
    ps = np.array([p for p in sorted(D) if F.POST_SWAP_P - 1e-9 <= p <= p_now + 1e-9])
    y = np.array([D[p] for p in ps])
    X = np.column_stack([np.ones_like(ps), ps - p_now])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    res = y - X @ beta
    s2 = res @ res / max(1, len(y) - 2)
    return float(beta[0]), float(np.sqrt(s2 * np.linalg.inv(X.T @ X)[0, 0])), float(np.sqrt(s2))


def dd_loo(dg, p_now, exclude):
    """其余 rung 的 ΔD = D(100%) − D(p_now):均值、标准差(下限 0.02)、个数。"""
    v = np.array([D[1.0] - np.interp(p_now, [0.55, 0.60], [D[0.55], D[0.60]])
                  for l, ws in dg.items() for w, D in ws.items()
                  if (l, w) != exclude and all(x in D for x in (0.55, 0.60, 1.0))])
    return float(v.mean()), float(max(v.std(ddof=1), F.DD_SD_FLOOR)), len(v)


def ptb_exposure_stats(long):
    """hero 的 D = PTB − 15 子集:第一次曝光前后相对线性趋势(35 个干净点)的平均残差,以及曝光前 30 个点自身的水平和斜率。"""
    h = long[long.category == "hero"].drop_duplicates(["step", "key"])
    w = h.pivot_table(index="step", columns="key", values="value").sort_index()
    sub = [c for c in w.columns if c.startswith("eval_dropless/paloma/") and c.endswith("-llama3/loss")]
    D = (w[F.PTB_KEY] - w[[c for c in sub if "/ptb-" not in c]].mean(axis=1)).dropna()
    D = D[D.index >= 116_999]
    clean = D[~D.index.isin(F.CONTAM_STEPS)]
    p = (clean.index.values + 1) / F.HERO_STEPS
    c = np.polyfit(p, clean.values, 1)
    r = clean - np.polyval(c, p)
    pre = clean[clean.index < F.CONTAM_STEPS[0]]
    cp = np.polyfit((pre.index.values + 1) / F.HERO_STEPS, pre.values, 1)
    pre_r = pre.values - np.polyval(cp, (pre.index.values + 1) / F.HERO_STEPS)
    p_now = p[-1]
    contam = D[D.index.isin(F.CONTAM_STEPS)] - np.polyval(c, (np.array(F.CONTAM_STEPS) + 1) / F.HERO_STEPS)
    return {"n_clean": int(len(clean)), "clean_resid_min": float(r.min()), "clean_resid_max": float(r.max()),
            "contaminated_resid": {int(k): float(v) for k, v in contam.items()},
            "post_exposure_mean_resid": float(r[r.index > F.CONTAM_STEPS[-1]].mean()),
            "n_post": int((r.index > F.CONTAM_STEPS[-1]).sum()),
            "post_exposure_mean_level": float(clean[clean.index > F.CONTAM_STEPS[-1]].mean()),
            "pre_exposure_slope": float(cp[0]), "pre_exposure_trend_at_now": float(np.polyval(cp, p_now)),
            "n_pre": int(len(pre)), "pre_exposure_resid_sd": float(np.std(pre_r, ddof=2)),
            "pre_exposure_lag1_autocorr": float(np.corrcoef(pre_r[:-1], pre_r[1:])[0, 1]),
            "mean_191999_200999": float(clean.loc[[191_999, 194_999, 197_999, 200_999]].mean()),
            "linear_level_now": float(np.polyval(c, p_now))}


# ---------------- 回测 ----------------
def pseudo_cases_v2(lads):
    G, Hn = lads["GB200"], lads["H100-new"]
    return F.pseudo_cases(lads) + [
        ("长距离 H100-new: d1536 ← d512,d768", Hn, ["d512", "d768"], Hn, "d1536"),
        ("长距离 跨阶梯: GB200 d1536 ← H100-new d512,d768", Hn, ["d512", "d768"], G, "d1536"),
    ]


def backtest_v2(lads, target):
    rows = []
    for name, train_lad, widths, tgt_lad, tgt_w in pseudo_cases_v2(lads):
        tg = tgt_lad["grids"][target][tgt_w]
        if 1.0 not in tg:
            continue
        dist = tgt_lad["rungs"][tgt_w]["C_nolm"] / max_rung_c(train_lad, widths)
        for p_now in (0.55, 0.60):
            hp = np.array([p for p in sorted(tg) if p <= p_now + 1e-9])
            hy = np.array([tg[p] for p in hp])
            for conv in F.CONV_W:
                for E in F.e_grid(conv):
                    f = F.smooth(F.traj(train_lad["grids"][target], train_lad["rungs"], widths, conv, E,
                                        tgt_lad["rungs"][tgt_w][conv]))
                    an = F.anchor_fit(f, hp, hy, p_now)
                    rows.append({"case": name, "dist": dist, "p_now": p_now, "conv": conv, "E": E, "actual": tg[1.0],
                                 "f1": f(1.0), "a": an["a"], "b": an["b"], "se_a": an["se_a"], "se_b": an["se_b"], "s": an["s"]})
    if target == "macro16":  # 67B:单点锚(21%、63%),预测 80%;没有平滑,b = 0
        b67 = F.backtest(lads, "macro16")
        b67 = b67[b67.case.str.startswith("67B")].copy()
        b67["dist"] = 1.19e23 / 7.16e20  # 目标 C ÷ 最大锚点 C(#6044)
        rows += b67[["case", "dist", "p_now", "conv", "E", "actual", "f1", "a", "b", "se_a", "se_b", "s"]].to_dict("records")
    bt = pd.DataFrame(rows)
    bt["horizon"] = np.where(bt.case.str.startswith("67B"), 0.80 - bt.p_now, 1 - bt.p_now)
    return bt


def case_errors(bt, kappa):
    """每个 (案例, 锚点):对 E 和口径取平均后的误差(预测 − 实际),κ 取给定值。"""
    err = bt.f1 + bt.a + kappa * bt.b * bt.horizon - bt.actual
    return err.groupby([bt.case, bt.p_now]).mean(), bt.groupby([bt.case, bt.p_now]).dist.first()


def choose_kappa(bt15):
    tab = []
    for k in KAPPA_GRID:
        ce, _ = case_errors(bt15, k)
        tab.append({"kappa": float(k), "rms": float(np.sqrt(np.mean(ce.values ** 2))), "mean": float(ce.mean())})
    tab = pd.DataFrame(tab)
    kmax = float(tab[tab.rms <= KAPPA_TOL * tab.rms.min()].kappa.max())
    return tab, kmax


def error_trend(ce, dist):
    """误差对 log10(外推距离) 的线性趋势。标准误按案例聚类:同一案例两个锚点的误差高度相关。"""
    x = np.log10(dist.values.astype(float))
    y = ce.values.astype(float)
    X = np.column_stack([np.ones_like(x), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    res = y - X @ beta
    cases = np.array([c for c, _ in ce.index])
    bread = np.linalg.inv(X.T @ X)
    meat = np.zeros((2, 2))
    for c in np.unique(cases):
        g = X[cases == c].T @ res[cases == c]
        meat += np.outer(g, g)
    cov = bread @ meat @ bread
    G, N = len(np.unique(cases)), len(y)
    cov_cr1 = cov * (G / (G - 1)) * ((N - 1) / (N - 2))  # 小样本修正(CR1)
    # 另一种看法:每个案例先把两个锚点平均,再对 G 个点做普通回归
    cm = pd.DataFrame({"c": cases, "x": x, "y": y}).groupby("c").mean()
    Xc = np.column_stack([np.ones(len(cm)), cm.x.values])
    bc = np.linalg.lstsq(Xc, cm.y.values, rcond=None)[0]
    rc = cm.y.values - Xc @ bc
    cov_c = (rc @ rc / (len(cm) - 2)) * np.linalg.inv(Xc.T @ Xc)
    return {"intercept": float(beta[0]), "slope": float(beta[1]), "slope_se": float(np.sqrt(cov_cr1[1, 1])),
            "slope_se_cr0": float(np.sqrt(cov[1, 1])), "n_cases": int(G),
            "case_mean_slope": float(bc[1]), "case_mean_slope_se": float(np.sqrt(cov_c[1, 1])),
            "case_mean_t": float(bc[1] / np.sqrt(cov_c[1, 1]))}


def bias_cap(trend, dist):
    """外推距离 d 处的偏差上限 b(d) = max(0, 截距 + 斜率 · log10 d)。"""
    return np.maximum(0.0, trend["intercept"] + trend["slope"] * np.log10(dist))


def eps_pool(bt15, bt16, kappa, target, include_67b=True):
    """σ_ε:各案例两个锚点的误差合在一起求均方根(围绕 0,不减趋势,偏保守)。15 子集也算上 67B 的两个误差:
    67B 只有 16 子集的数据,但它是唯一的大规模 MoE,外推距离也最长。自由度 = 案例数 − 1。
    同一个误差池再拟合误差对 log10(外推距离) 的趋势,给出偏差上限 b(d)。"""
    ce15, d15 = case_errors(bt15, kappa)
    ce16, d16 = case_errors(bt16, kappa)
    is67 = np.array([c.startswith("67B") for c, _ in ce16.index])
    if target == "macro15":
        ce, dist = (pd.concat([ce15, ce16[is67]]), pd.concat([d15, d16[is67]])) if include_67b else (ce15, d15)
    else:
        ce, dist = (ce16, d16) if include_67b else (ce16[~is67], d16[~is67])
    n_cases = len({c for c, _ in ce.index})
    groups = {}
    for lo, hi, label in DIST_BINS:
        m = (dist.values >= lo) & (dist.values < hi)
        if m.any():
            groups[label] = {"n": int(m.sum()), "rms": float(np.sqrt(np.mean(ce.values[m] ** 2))), "mean": float(ce.values[m].mean())}
    return {"sigma": float(np.sqrt(np.mean(ce.values ** 2))), "mean": float(ce.mean()), "nu": n_cases - 1,
            "n_cases": n_cases, "n_errors": int(len(ce)), "by_distance": groups, "trend": error_trend(ce, dist),
            "case_errors": {f"{c} @{p}": float(v) for (c, p), v in ce.items()}}


def target_errors(bt15, bt16, lads, dg, kappa):
    """每个案例、每个锚点在三种目标上的点误差:15 子集、16 子集直接锚定、官方组合(15 子集 + D/16)。"""
    names = {id(v): k for k, v in lads.items()}
    tgt = {name: (names[id(tl)], tw) for name, _, _, tl, tw in pseudo_cases_v2(lads)}
    ce15, _ = case_errors(bt15, kappa)
    ce16, dist16 = case_errors(bt16, kappa)
    rows = []
    for (case, p_now), e16 in ce16.items():
        r = {"case": case, "p_now": p_now, "dist": float(dist16[(case, p_now)]), "err_macro16_direct": float(e16),
             "err_macro15": np.nan, "err_official_composite": np.nan}
        if not case.startswith("67B"):
            tn, tw = tgt[case]
            d_now, _, _ = d_now_fit(dg[tn][tw], p_now)
            dd, _, _ = dd_loo(dg, p_now, (tn, tw))
            sub = bt15[(bt15.case == case) & (bt15.p_now == p_now)]
            pred15 = float((sub.f1 + sub.a + kappa * sub.b * sub.horizon).mean())
            r["err_macro15"] = float(ce15[(case, p_now)])
            r["err_official_composite"] = pred15 + (d_now + dd) / 16 - lads[tn]["grids"]["macro16"][tw][1.0]
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------- 蒙特卡洛 ----------------
def mc_v2(rows, p_now, kappa_max, sigma_eps, nu, sigma_eval, n, rng, p_out=None, trend=None, bias_mode="uniform",
          kappa_fixed=None):
    """bias_mode:"uniform" 时 β ~ U[0, b(d)],"full" 时 β = b(d),"none" 或没有 trend 时 β = 0。
    误差记为"预测 − 实际",所以从预测里减掉 β。"""
    w = np.array([r["w"] for r in rows], float)
    w /= w.sum()
    idx = rng.choice(len(rows), size=n, p=w)
    f1 = np.array([rows[i]["f1"] for i in idx])
    a = np.array([rows[i]["a"] for i in idx]) + rng.normal(size=n) * np.array([rows[i]["se_a"] for i in idx])
    b = np.array([rows[i]["b"] for i in idx]) + rng.normal(size=n) * np.array([rows[i]["se_b"] for i in idx])
    kappa = rng.uniform(0, kappa_max, size=n) if kappa_fixed is None else np.full(n, float(kappa_fixed))
    z = rng.standard_t(nu, size=n) if nu else rng.normal(size=n)
    u = rng.uniform(size=n)
    if trend is None or bias_mode == "none":
        beta = np.zeros(n)
    else:
        cap = bias_cap(trend, np.array([rows[i]["dist"] for i in idx]))
        beta = cap * (u if bias_mode == "uniform" else 1.0)
    eps = z * sigma_eps - beta
    end = f1 + a + kappa * b * (1 - p_now) + eps + rng.normal(size=n) * sigma_eval
    if p_out is None:
        return end, None
    tr = {}
    for p in p_out:
        fp = np.array([rows[i]["traj"][p] for i in idx])
        frac = max(0.0, (p - p_now) / (1 - p_now))
        tr[p] = fp + a + kappa * b * (p - p_now) + eps * frac + rng.normal(size=n) * sigma_eval
    return end, tr


def ptb_autocorr(long, p_now):
    """hero 的 D 序列(35 个干净点)对线性趋势的残差的一阶自相关,以及据此放宽后的 D_now 标准误倍数。"""
    h = long[long.category == "hero"].drop_duplicates(["step", "key"])
    w = h.pivot_table(index="step", columns="key", values="value").sort_index()
    sub = [c for c in w.columns if c.startswith("eval_dropless/paloma/") and c.endswith("-llama3/loss")]
    D = (w[F.PTB_KEY] - w[[c for c in sub if "/ptb-" not in c]].mean(axis=1)).dropna()
    D = D[(D.index >= 116_999) & ~D.index.isin(F.CONTAM_STEPS)]
    p = (D.index.values + 1) / F.HERO_STEPS
    r = D.values - np.polyval(np.polyfit(p, D.values, 1), p)
    rho = float(np.corrcoef(r[:-1], r[1:])[0, 1])
    return {"lag1_autocorr": rho, "se_factor": float(np.sqrt((1 + rho) / (1 - rho)))}


def sample_D_v2(pt, n, p_now, p_out, rng, allowance="v2", end_p=1.0, complete_decay=False):
    """D = PTB − 15 子集。D_now 的标准误用 pt["se_D_now_v2"](按自相关放宽)。allowance:
    "v2":r2 ~ U[−0.06, 0],只计入第 362,280 步及以后的点;
    "v1":U[−0.06, 0],按剩余进度线性计入(v1 的做法,标准误也用 v1 的);"none":不加。"""
    se = pt["se_D_now"] if allowance == "v1" else pt["se_D_now_v2"]
    d_now = pt["D_now"] + rng.normal(size=n) * se
    dd = rng.normal(pt["dD_mean"], pt["dD_sd_used"], size=n)
    r_v1 = PTB_V1[0] + rng.uniform(size=n) * (PTB_V1[1] - PTB_V1[0])
    r2 = PTB_R2[0] + rng.uniform(size=n) * (PTB_R2[1] - PTB_R2[0])

    def allow(p, frac):
        if allowance == "none":
            return 0.0
        if allowance == "v1":
            return r_v1 * frac
        return r2 if p >= P_EXPO2 - 1e-12 else 0.0

    end = d_now + dd + allow(end_p, 1.0) + rng.normal(size=n) * pt["sd_noise"]
    traj = {}
    for p in p_out:
        frac = 1.0 if complete_decay else max(0.0, (p - p_now) / (1 - p_now))
        traj[p] = d_now + dd * frac + allow(p, frac) + rng.normal(size=n) * pt["sd_noise"]
    return end, traj


def coverage_v2(bt15, bt16, lads, dg, sev16, n=20_000):
    """嵌套留一:留出一个案例(两个锚点一起),κ_max、σ_ε 和距离趋势只用其余案例来定,再用和 hero 相同的蒙特卡洛给区间。"""
    names = {id(v): k for k, v in lads.items()}
    tgt = {name: (names[id(tl)], tw) for name, _, _, tl, tw in pseudo_cases_v2(lads)}
    rng = np.random.default_rng(SEED + 7)
    rows = []
    for case in bt16.case.unique():
        o15, o16 = bt15[bt15.case != case], bt16[bt16.case != case]
        _, kmax = choose_kappa(o15)
        e15, e16 = eps_pool(o15, o16, kmax / 2, "macro15"), eps_pool(o15, o16, kmax / 2, "macro16")

        for p_now in sorted(bt16[bt16.case == case].p_now.unique()):
            def interval(bt, eps, s_eval=None, se_a=None):
                sub = bt[(bt.case == case) & (bt.p_now == p_now)]
                tab = [{"f1": r.f1, "a": r.a, "b": r.b, "se_a": r.se_a if se_a is None else se_a, "se_b": r.se_b,
                        "dist": r.dist, "w": 1.0} for r in sub.itertuples()]
                s = float(sub.s.mean()) if s_eval is None else s_eval
                end, _ = mc_v2(tab, 1 - float(sub.horizon.iloc[0]), kmax, eps["sigma"], eps["nu"], s, n, rng, trend=eps["trend"])
                return end, float(sub.actual.iloc[0]), eps
            out = {}
            if case.startswith("67B"):  # 单点锚:锚点和终点各带一次评测噪声
                out["macro16_direct"] = interval(bt16, e16, s_eval=sev16, se_a=sev16)
            else:
                end15, act15, _ = interval(bt15, e15)
                out["macro15"] = (end15, act15, e15)
                out["macro16_direct"] = interval(bt16, e16)
                tn, tw = tgt[case]
                d_now, se_d, sd_d = d_now_fit(dg[tn][tw], p_now)
                dd_m, dd_s, _ = dd_loo(dg, p_now, (tn, tw))
                d_end = d_now + rng.normal(size=n) * se_d + rng.normal(dd_m, dd_s, size=n) + rng.normal(size=n) * sd_d
                out["official_composite"] = (end15 + d_end / 16, lads[tn]["grids"]["macro16"][tw][1.0], e15)
            for t, (end, act, eps) in out.items():
                q = F.quantiles(end)
                rows.append({"target": t, "case": case, "p_now": p_now, "kappa_max_loo": kmax, "sigma_eps_loo": eps["sigma"],
                             "nu_loo": eps["nu"], "trend_slope_loo": eps["trend"]["slope"], "q025": q["q025"], "q10": q["q10"],
                             "median": q["median"], "q90": q["q90"], "q975": q["q975"], "actual": act,
                             "pit": float(np.mean(end <= act)),
                             "in80": bool(q["q10"] <= act <= q["q90"]), "in95": bool(q["q025"] <= act <= q["q975"])})
    return pd.DataFrame(rows)


# ---------------- hero ----------------
def hero_rows(lads, target, hp, hy, p_now, p_out, spec=None):
    """spec = {阶梯: (权重, 用到的 rung)};默认两条阶梯、全部 rung,GB200 的 d2048 用三套经验剖面。"""
    spec = spec or {k: (w, list(lads[k]["rungs"])) for k, w in F.LADDER_W.items()}
    profs = d2048_profiles(lads, target)
    rows = []
    for lad_name, (lw, widths) in spec.items():
        lad = lads[lad_name]
        variants = list(profs.items()) if "d2048" in widths else [(None, None)]
        for conv, cw in F.CONV_W.items():
            Es = F.e_grid(conv, step=0.01)
            for pname, prof in variants:
                grids = dict(lad["grids"][target])
                if prof is not None:
                    grids["d2048"] = extend_d2048_profile(grids["d2048"], prof)
                for E in Es:
                    f = F.smooth(F.traj(grids, lad["rungs"], widths, conv, E, F.HERO[conv]))
                    an = F.anchor_fit(f, hp, hy, p_now)
                    rows.append({"ladder": lad_name, "conv": conv, "E": float(E), "profile": pname, "f1": float(f(1.0)),
                                 "a": an["a"], "b": an["b"], "se_a": an["se_a"], "se_b": an["se_b"], "s": an["s"],
                                 "traj": {p: float(f(p)) for p in p_out}, "w": lw * cw / (len(Es) * len(variants)),
                                 "dist": F.HERO["C_nolm"] / max_rung_c(lad, widths)})
    return rows


def distance_checks(lads, hp, hy, p_now):
    """hero 内部的距离检查:同一条阶梯去掉较大的 rung、外推得更远时,15 子集的中心值怎么变。
    中心值 = 各行 f(1) + a + 0.1·b·(1 − p_now) 的加权平均(κ 取先验均值,不含噪声和距离偏差项)。"""
    sets = {"GB200": [["d768", "d1024", "d1536", "d2048"], ["d768", "d1024", "d1536"], ["d768", "d1024"]],
            "H100-new": [["d512", "d768", "d1024", "d1536"], ["d512", "d768", "d1024"], ["d512", "d768"]]}
    out = []
    for lad, subsets in sets.items():
        for ws in subsets:
            rows = hero_rows(lads, "macro15", hp, hy, p_now, [], {lad: (1.0, ws)})
            w = np.array([r["w"] for r in rows])
            x = np.array([r["f1"] + r["a"] + 0.1 * r["b"] * (1 - p_now) for r in rows])
            out.append({"ladder": lad, "rungs": ",".join(ws), "extrapolation": rows[0]["dist"],
                        "center_macro15": float(np.average(x, weights=w))})
    return out


def official(rows, p_now, pt, sev15, kmax, eps, seed, n=40_000, allowance="v2", **kw):
    rng = np.random.default_rng(seed)
    e15, _ = mc_v2(rows, p_now, kmax, eps["sigma"], eps["nu"], sev15, n, rng, trend=eps.get("trend"), **kw)
    de, _ = sample_D_v2(pt, n, p_now, [], rng, allowance=allowance)
    return e15, e15 + de / 16


def bridge(lads, hp15, hy15, p_now, sev15, pt, kmax, eps15, rows15):
    """从 v1 到 v2 逐项替换,看每一项改动让官方 macro 的预测怎么变(同一份数据、同一组随机数)。"""
    v1rows = F.hero_table(lads, "macro15", hp15, hy15, p_now)
    sig1 = float(np.sqrt(np.mean(F.case_errors(F.backtest(lads, "macro15")).values ** 2)))
    v1eps = {"sigma": sig1, "nu": None, "trend": None}
    eps_no_trend = dict(eps15, trend=None)
    steps = [
        ("v1 method, re-run on the same data", v1rows, 1.0, v1eps, "v1"),
        ("+ drift prior kappa ~ U[0, kappa_max]", v1rows, kmax, v1eps, "v1"),
        ("+ empirical d2048 profiles", rows15, kmax, v1eps, "v1"),
        ("+ method error pooled with the 67B, Student-t", rows15, kmax, eps_no_trend, "v1"),
        ("+ distance-bias allowance", rows15, kmax, eps15, "v1"),
        ("+ PTB: allowance timed at the 2nd exposure, wider D_now SE (= v2)", rows15, kmax, eps15, "v2"),
    ]
    out = []
    for name, rows, km, eps, allowance in steps:
        e15, e16 = official(rows, p_now, pt, sev15, km, eps, SEED + 100, n=N_MC, allowance=allowance)
        q, q15 = F.quantiles(e16), F.quantiles(e15)
        out.append({"step": name, "median": q["median"], "q10": q["q10"], "q90": q["q90"], "q025": q["q025"],
                    "q975": q["q975"], "macro15_median": q15["median"], "macro15_q10": q15["q10"], "macro15_q90": q15["q90"]})
    return out


def shortened(lads, rows15, p_now, pt, sev15, kmax, eps15):
    """训练被缩短到 r·T 并在新终点完成衰减:终点 = 锚定终点 + 100% 拟合在 r·C 与 C 处之差;
    漂移按缩短后的剩余进度算;PTB 差值的变化按完整衰减计;第二次曝光之前结束的不计 r2。"""
    profs = d2048_profiles(lads, "macro15")
    out = {}
    for ratio in (0.80, 0.85, 0.90, 0.95, 1.00):
        mod = []
        for r in rows15:
            lad = lads[r["ladder"]]
            grids = dict(lad["grids"]["macro15"])
            if r["profile"] is not None:
                grids["d2048"] = extend_d2048_profile(grids["d2048"], profs[r["profile"]])
            pts = [(lad["rungs"][w][r["conv"]], grids[w][1.0]) for w in lad["rungs"] if 1.0 in grids[w]]
            c = np.array([x[0] for x in pts])
            y = np.array([x[1] for x in pts])
            s_, i_ = np.polyfit(np.log(c), np.log(y - r["E"]), 1)
            law = lambda C: r["E"] + np.exp(i_) * C ** s_
            m = dict(r)
            m["f1"] = r["f1"] + float(law(F.HERO[r["conv"]] * ratio) - law(F.HERO[r["conv"]]))
            m["b"] = r["b"] * (ratio - p_now) / (1 - p_now)
            m["se_b"] = r["se_b"] * (ratio - p_now) / (1 - p_now)
            mod.append(m)
        last = int(round(ratio * F.HERO_STEPS)) - 1
        rng = np.random.default_rng(SEED + 200)
        e15, _ = mc_v2(mod, p_now, kmax, eps15["sigma"], eps15["nu"], sev15, 40_000, rng, trend=eps15["trend"])
        de, _ = sample_D_v2(pt, 40_000, p_now, [], rng, end_p=(last + 1) / F.HERO_STEPS, complete_decay=True)
        out[f"{ratio:.2f}"] = {"final_step": last, "official_macro16": F.quantiles(e15 + de / 16), "macro15": F.quantiles(e15),
                               "in_acute_window_of_exposure_2": bool(EXPO2[0] <= last <= EXPO2[1] + ACUTE_STEPS)}
    return out


# ---------------- 输出 ----------------
def write_traj_csv(tr, path, p_now):
    with open(path, "w") as f:
        f.write("fraction,step,q025,q10,median,q90,q975\n")
        for p in sorted(tr):
            if p + 1e-9 < p_now:
                continue
            q = F.quantiles(tr[p])
            f.write(f"{p:.6f},{int(round(p * F.HERO_STEPS)) - 1},{q['q025']:.5f},{q['q10']:.5f},{q['median']:.5f},{q['q90']:.5f},{q['q975']:.5f}\n")


def plot_en(tr16, tr15, long, path, step):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, tr, key, title in ((axes[0], tr16, "macro16", "eval_dropless/paloma/macro_loss (16 subsets, primary)"),
                               (axes[1], tr15, "macro15", "15-subset macro without PTB (companion)")):
        hp, hy = F.hero_series(long, key)
        ps = np.array(sorted(tr))
        qs = [F.quantiles(tr[p]) for p in ps]
        ax.plot(hp * 100, hy, "-", color="black", lw=1.4, label="hero, observed")
        ax.fill_between(ps * 100, [q["q025"] for q in qs], [q["q975"] for q in qs], color="#1f77b4", alpha=0.18, label="95% interval")
        ax.fill_between(ps * 100, [q["q10"] for q in qs], [q["q90"] for q in qs], color="#1f77b4", alpha=0.35, label="80% interval")
        ax.plot(ps * 100, [q["median"] for q in qs], "-", color="#1f77b4", lw=2, label="median")
        if key == "macro16":
            for (name, v), col in zip(TEAM_EN.items(), ("crimson", "darkorange", "seagreen")):
                ax.plot(100, v, "D", color=col, ms=8, label=f"team {name}: {v:.3f}")
            ax.axvspan(EXPO2[0] / F.HERO_STEPS * 100, (EXPO2[1] + ACUTE_STEPS) / F.HERO_STEPS * 100, color="grey", alpha=0.12,
                       label="2nd PTB exposure, acute window")
        lo = min(hy[hp > 0.25].min(), min(q["q025"] for q in qs)) - 0.02
        ax.set_ylim(lo, hy[hp > 0.25].max() + 0.01)
        ax.set_xlim(25, 102)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("training progress (%)")
        ax.set_ylabel("dropless Paloma loss (4K packing)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8.5, loc="upper right")
    fig.suptitle(f"Anchored forecast v2 of the 535B-A23B hero run's final loss (data through step {step:,})")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if math.isnan(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def main(snap, outdir):
    os.makedirs(outdir, exist_ok=True)
    long, runs = load(snap)
    long = dedupe_latest(long)
    lads = F.build_ladders(long, runs)
    dg = d_grids(long, runs, lads)

    # 回测:选 κ_max,估 σ_ε 和距离趋势,三种目标的点误差,嵌套留一校准
    bt15, bt16 = backtest_v2(lads, "macro15"), backtest_v2(lads, "macro16")
    ktab, kmax = choose_kappa(bt15)
    kmean = kmax / 2
    eps15, eps16 = eps_pool(bt15, bt16, kmean, "macro15"), eps_pool(bt15, bt16, kmean, "macro16")
    eps15_no67 = eps_pool(bt15, bt16, kmean, "macro15", include_67b=False)
    terr = target_errors(bt15, bt16, lads, dg, kmean)
    hp15, hy15 = F.hero_series(long, "macro15")
    hp16, hy16 = F.hero_series(long, "macro16")
    sev15, sev16 = F.detrended_sd(hp15, hy15), F.detrended_sd(hp16, hy16)
    cov = coverage_v2(bt15, bt16, lads, dg, sev16)
    pd.concat([bt15.assign(target="macro15"), bt16.assign(target="macro16")]).to_csv(os.path.join(outdir, "backtest_v2.csv"), index=False)
    terr.to_csv(os.path.join(outdir, "backtest_errors_by_target.csv"), index=False)
    cov.to_csv(os.path.join(outdir, "coverage_v2.csv"), index=False)
    ktab.to_csv(os.path.join(outdir, "kappa_backtest.csv"), index=False)

    # hero
    p_now = float(hp15[-1])
    start = math.ceil(p_now * 100 - 1e-9) / 100
    p_out = sorted(set([round(x, 2) for x in np.arange(start, 1.0 + 1e-9, 0.01)] + [1.0]
                       + [EXPO2[0] / F.HERO_STEPS, P_EXPO2]))  # 第 362,279 步(曝光前)和第 362,280 步(曝光开始)
    rows15 = hero_rows(lads, "macro15", hp15, hy15, p_now, p_out)
    pt = F.ptb_gap_model(long, runs, lads, p_now)
    pac = ptb_autocorr(long, p_now)
    pt["se_D_now_v2"] = pt["se_D_now"] * pac["se_factor"]
    rng = np.random.default_rng(SEED)
    end15, tr15 = mc_v2(rows15, p_now, kmax, eps15["sigma"], eps15["nu"], sev15, N_MC, rng, p_out, trend=eps15["trend"])
    d_end, d_tr = sample_D_v2(pt, N_MC, p_now, p_out, rng)
    end16 = end15 + d_end / 16
    tr16 = {p: tr15[p] + d_tr[p] / 16 for p in p_out}
    rows16d = hero_rows(lads, "macro16", hp16, hy16, p_now, p_out)
    end16d, _ = mc_v2(rows16d, p_now, kmax, eps16["sigma"], eps16["nu"], sev16, N_MC, np.random.default_rng(SEED + 1),
                      trend=eps16["trend"])

    # 敏感性:每行只改一处,其余同 v2(同一组随机数,4 万次抽样)
    allw = {k: list(lads[k]["rungs"]) for k in lads}
    sens = {}
    base_kw = dict(p_now=p_now, pt=pt, sev15=sev15, seed=SEED + 300)
    sens["v2 baseline (40,000 draws, same random numbers as the rows below)"] = official(rows15, kmax=kmax, eps=eps15, **base_kw)[1]
    sens["kappa = 0 (no drift)"] = official(rows15, kmax=kmax, eps=eps15, kappa_fixed=0.0, **base_kw)[1]
    sens["kappa ~ U[0, 1] (v1 prior)"] = official(rows15, kmax=1.0, eps=eps15, **base_kw)[1]
    specs = {
        "GB200 ladder only (337x)": {"GB200": (1.0, allw["GB200"])},
        "H100 new-mix ladder only (1,890x)": {"H100-new": (1.0, allw["H100-new"])},
        "GB200 without d2048 (1,890x)": {"GB200": (1.0, ["d768", "d1024", "d1536"])},
        "H100 without d1536 (15,126x)": {"H100-new": (1.0, ["d512", "d768", "d1024"])},
        "ladder weights 50/50": {"GB200": (0.5, allw["GB200"]), "H100-new": (0.5, allw["H100-new"])},
    }
    for name, spec in specs.items():
        rows = hero_rows(lads, "macro15", hp15, hy15, p_now, [], spec)
        sens[name] = official(rows, kmax=kmax, eps=eps15, **base_kw)[1]
    sens["Normal instead of Student-t"] = official(rows15, kmax=kmax, eps=dict(eps15, nu=None), **base_kw)[1]
    sens["method error and distance trend without the 67B"] = official(rows15, kmax=kmax, eps=eps15_no67, **base_kw)[1]
    sens["no distance-bias allowance"] = official(rows15, kmax=kmax, eps=eps15, bias_mode="none", **base_kw)[1]
    sens["full distance trend (beta = b(d))"] = official(rows15, kmax=kmax, eps=eps15, bias_mode="full", **base_kw)[1]
    sens["PTB allowance as in v1 (U[-0.06, 0], phased in)"] = official(rows15, kmax=kmax, eps=eps15, allowance="v1", **base_kw)[1]
    sens["no PTB allowance"] = official(rows15, kmax=kmax, eps=eps15, allowance="none", **base_kw)[1]

    # v1 的稳健性检验(平滑方式、锚定窗口)在 v2 下重做
    def patched_rows(deg=None, raw=False, p_min=None):
        old = (F.SMOOTH_DEG, F.smooth)
        if deg is not None:
            F.SMOOTH_DEG = deg
        if raw:  # 不平滑:逐 5% 拟合的点之间线性插值
            F.smooth = lambda ys: (lambda p: np.interp(p, sorted(ys), [ys[k] for k in sorted(ys)]))
        keep = hp15 >= (p_min or 0.0)
        try:
            return hero_rows(lads, "macro15", hp15[keep], hy15[keep], p_now, [])
        finally:
            F.SMOOTH_DEG, F.smooth = old
    robust = {
        "quadratic smoothing": dict(deg=2),
        "quartic smoothing": dict(deg=4),
        "no smoothing (linear interpolation of the per-5% fits)": dict(raw=True),
        "anchor window from 45%": dict(p_min=0.45),
    }
    for name, kw in robust.items():
        sens[name] = official(patched_rows(**kw), kmax=kmax, eps=eps15, **base_kw)[1]
    sens = {k: F.quantiles(v) for k, v in sens.items()}

    br = bridge(lads, hp15, hy15, p_now, sev15, pt, kmax, eps15, rows15)
    short = shortened(lads, rows15, p_now, pt, sev15, kmax, eps15)
    by_ladder = {lad: F.quantiles(official([r for r in rows15 if r["ladder"] == lad], kmax=kmax, eps=eps15, **base_kw)[1])
                 for lad in F.LADDER_W}
    by_ladder_no_bias = {lad: F.quantiles(official([r for r in rows15 if r["ladder"] == lad], kmax=kmax, eps=eps15,
                                                   bias_mode="none", **base_kw)[1]) for lad in F.LADDER_W}
    drop_no_bias = {}
    for name in ("GB200 ladder only (337x)", "GB200 without d2048 (1,890x)", "H100 new-mix ladder only (1,890x)",
                 "H100 without d1536 (15,126x)"):
        rows = hero_rows(lads, "macro15", hp15, hy15, p_now, [], specs[name])
        drop_no_bias[name] = F.quantiles(official(rows, kmax=kmax, eps=eps15, bias_mode="none", **base_kw)[1])
    dchecks = distance_checks(lads, hp15, hy15, p_now)

    step = int(round(p_now * F.HERO_STEPS)) - 1
    write_traj_csv(tr16, os.path.join(outdir, "trajectory_official_macro16_v2.csv"), p_now)
    write_traj_csv(tr15, os.path.join(outdir, "trajectory_macro15_v2.csv"), p_now)
    np.savez_compressed(os.path.join(outdir, "end_samples_v2.npz"), macro16=end16.astype(np.float32), macro15=end15.astype(np.float32))
    plot_en(tr16, tr15, long, os.path.join(outdir, "forecast_v2_en.png"), step)

    covsum = {t: {"in80": f"{int(g.in80.sum())}/{len(g)}", "in95": f"{int(g.in95.sum())}/{len(g)}",
                  "mean_pit": float(g.pit.mean())} for t, g in cov.groupby("target")}
    hero_dist = {"GB200": F.HERO["C_nolm"] / max_rung_c(lads["GB200"], allw["GB200"]),
                 "H100-new": F.HERO["C_nolm"] / max_rung_c(lads["H100-new"], allw["H100-new"])}
    man = json.load(open(os.path.join(snap, "manifest.json")))
    summary = {
        "version": "v2", "data_through_step": step, "fraction": p_now,
        "snapshot": os.path.relpath(snap, os.path.dirname(HERE)), "snapshot_fetched_utc": man["finished_utc"],
        "official_macro16": {"end": F.quantiles(end16),
                             "probs": {"P(<=2.039)": float(np.mean(end16 <= 2.039)), "P(<=2.02685)": float(np.mean(end16 <= 2.02685)),
                                       "P(>=2.07)": float(np.mean(end16 >= 2.07)), "P(>=2.100)": float(np.mean(end16 >= 2.100))},
                             "team_percentile": {k: float(np.mean(end16 <= v)) for k, v in TEAM_EN.items()}},
        "macro15": {"end": F.quantiles(end15)},
        "direct_macro16_variant": F.quantiles(end16d),
        "by_ladder_official": by_ladder, "by_ladder_official_no_distance_allowance": by_ladder_no_bias,
        "distance_checks_within_hero": dchecks, "drop_rung_official_no_distance_allowance": drop_no_bias,
        "drift": {"kappa_max": kmax, "kappa_tolerance": KAPPA_TOL},
        "method_error": {"macro15_incl_67b": {k: v for k, v in eps15.items() if k != "case_errors"},
                         "macro16_direct": {k: v for k, v in eps16.items() if k != "case_errors"}},
        "distance_bias_cap_at_hero": {lad: float(bias_cap(eps15["trend"], d)) for lad, d in hero_dist.items()},
        "eval_noise": {"macro15": sev15, "macro16": sev16},
        "hero_extrapolation": hero_dist,
        "coverage_nested_loo": covsum,
        "backtest_rms_ladder_cases": {c: float(np.sqrt(np.mean(terr.loc[terr.err_macro15.notna(), c] ** 2)))
                                      for c in ("err_macro15", "err_official_composite", "err_macro16_direct")},
        "bridge_v1_to_v2": br, "sensitivity_official": sens, "shortened_horizon": short,
        "ptb_model": {k: v for k, v in pt.items() if k != "rungs"},
        "ptb_allowance": {"r2_from_step": EXPO2[0], "r2": list(PTB_R2), "exposure_2_steps": list(EXPO2),
                          "acute_window_steps_after_exposure": ACUTE_STEPS},
        "ptb_D_now_autocorrelation": dict(pac, se_D_now=pt["se_D_now"], se_D_now_v2=pt["se_D_now_v2"]),
        "ptb_exposure_evidence": ptb_exposure_stats(long),
        "d2048_profiles": {t: d2048_profiles(lads, t) for t in F.TARGETS},
    }
    json.dump(to_jsonable(summary), open(os.path.join(outdir, "forecast_summary_v2.json"), "w"), ensure_ascii=False, indent=1)
    json.dump(to_jsonable({"macro15_incl_67b": eps15, "macro16_direct": eps16, "macro15_without_67b": eps15_no67}),
              open(os.path.join(outdir, "method_error_v2.json"), "w"), ensure_ascii=False, indent=1)

    e = summary["official_macro16"]["end"]
    print(f"== v2 == 数据截至第 {step:,} 步(p={p_now:.4f})")
    print(f"κ_max = {kmax}")
    for nm, ep in (("15 子集 + 67B", eps15), ("16 子集直接", eps16), ("15 子集不含 67B", eps15_no67)):
        t = ep["trend"]
        print(f"σ_ε {nm}: {ep['sigma']:.4f}(均值 {ep['mean']:+.4f},{ep['n_cases']} 个案例,ν = {ep['nu']});"
              f"趋势 {t['slope']:+.4f}/十倍(SE {t['slope_se']:.4f});b(337) {float(bias_cap(t, 337.2)):.4f},b(1890) {float(bias_cap(t, 1890.2)):.4f}")
    print(f"官方 macro:中位数 {e['median']:.4f};80% [{e['q10']:.4f}, {e['q90']:.4f}];95% [{e['q025']:.4f}, {e['q975']:.4f}]")
    m = summary["macro15"]["end"]
    print(f"15 子集:中位数 {m['median']:.4f};80% [{m['q10']:.4f}, {m['q90']:.4f}];95% [{m['q025']:.4f}, {m['q975']:.4f}]")
    print("概率:", {k: round(v, 4) for k, v in summary["official_macro16"]["probs"].items()})
    print("按阶梯:", {k: f"{v['median']:.4f} [{v['q10']:.4f}, {v['q90']:.4f}]" for k, v in by_ladder.items()})
    print("直接锚定对照:", f"{summary['direct_macro16_variant']['median']:.4f}")
    print("按阶梯(不加距离偏差):", {k: f"{v['median']:.4f}" for k, v in by_ladder_no_bias.items()})
    print("去掉最大 rung(不加距离偏差):", {k: round(v["median"], 4) for k, v in drop_no_bias.items()})
    print("PTB 自相关:", {k: round(v, 4) for k, v in pac.items()}, "se_D_now", round(pt["se_D_now"], 4), "→", round(pt["se_D_now_v2"], 4))
    print("hero 内部距离检查:", [(d["ladder"], d["rungs"], f"{d['extrapolation']:,.0f}x", round(d["center_macro15"], 4)) for d in dchecks])
    print("回测 RMS(8 个阶梯案例):", {k: round(v, 4) for k, v in summary["backtest_rms_ladder_cases"].items()})
    print("嵌套留一:", covsum)
    print("v1 → v2:")
    for b in br:
        print(f"   {b['median']:.4f} [{b['q10']:.4f}, {b['q90']:.4f}]  15 子集 {b['macro15_median']:.4f}  {b['step']}")
    print("敏感性:")
    for k, v in sens.items():
        print(f"   {v['median']:.4f} [{v['q10']:.4f}, {v['q90']:.4f}]  {k}")
    print("缩短:", {k: f"{v['official_macro16']['median']:.4f}{' (急性期)' if v['in_acute_window_of_exposure_2'] else ''}" for k, v in short.items()})
    print("PTB 证据:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary["ptb_exposure_evidence"].items()})


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_only(outdir, prereg):
    root = os.path.dirname(HERE)
    files = [os.path.join(HERE, f) for f in CODE_FILES]
    files += [os.path.join(outdir, f) for f in sorted(os.listdir(outdir)) if f.endswith((".json", ".csv", ".png", ".npz"))
              and f != "freeze_manifest_v2.json"]
    files.append(os.path.abspath(prereg))
    summ = json.load(open(os.path.join(outdir, "forecast_summary_v2.json")))
    files.append(os.path.join(root, summ["snapshot"], "manifest.json"))
    out = {"frozen_files": {os.path.relpath(p, root): sha256(p) for p in files}}
    json.dump(out, open(os.path.join(outdir, "freeze_manifest_v2.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "--hash-only":
        hash_only(sys.argv[2], sys.argv[3])
    else:
        main(sys.argv[1], sys.argv[2])
