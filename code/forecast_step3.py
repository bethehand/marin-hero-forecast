#!/usr/bin/env python3
"""第 3 步:用锚定法给 535B 的终点 loss 做带置信区间的预测。决定 D1–D6 见 03_第3步设计草案.md。

  D1 目标:官方 16 子集 macro(主)和去掉 PTB 的 15 子集 macro(伴随),第 390,251 步,4K 评测打包。
  D2 方法:锚定法。终点 = 预测曲线在终点的值 + hero 相对预测曲线的偏移 + 漂移 + 方法误差 + 评测噪声。
  D3 漂移:用换配比之后的残差估计斜率 b,乘以"持续系数" κ ~ U[0,1] 外推到终点。
  D4 阶梯:GB200 权重 0.75,H100 新配比权重 0.25。
  D5 E 的先验:不含 lm_head 取 U[1.3, 1.7],含 lm_head 取 U[1.5, 1.9];两种口径各占一半。
  D6 报 80% 和 95% 区间;打分规则写在结果文档里,训练结束后用 CRPS 和团队的点预测比。

预测曲线:每 5% 进度单独做 log 空间拟合(E 固定,和预注册同一种做法),
再在 30%–100% 上用三次多项式平滑,减少逐点拟合带来的抖动。
方法误差 ε:用"伪 hero"回测得到。伪 hero 包括同阶梯、跨阶梯的最大一级,以及 67B。

用法:
  python forecast_step3.py ../data/snapshot_2026-10-09 ../results/step3
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from explore_step3 import HERO, gb200_ladder, h100_ladder, lm_flops
from reproduce import load

HERO_STEPS = 390_251
TARGETS = ("macro16", "macro15")
E_PRIOR = {"C_nolm": (1.30, 1.70), "C_lm": (1.50, 1.90)}  # D5
LADDER_W = {"GB200": 0.75, "H100-new": 0.25}  # D4
CONV_W = {"C_nolm": 0.5, "C_lm": 0.5}
D2048_CORR = (0.003, 0.006)  # d2048 在 80% 以后线性外推的修正:团队用 0.005,我们在 GB200 已完成的 rung 上实测约 0.003
POST_SWAP_P = 0.299  # 第 108,000 步(27.7%)换配比之后;取 ≥30% 的评测点
FRACS = [round(0.05 * i, 2) for i in range(1, 21)]
P_OUT = [round(0.60 + 0.05 * i, 2) for i in range(9)]  # 60%, 65%, …, 100%
SMOOTH_FROM, SMOOTH_DEG = 0.30, 3
N_MC = 60_000
RNG = np.random.default_rng(20261009)
# 官方 16 子集 macro = 15 子集 macro + D/16,D = PTB − 15 子集 macro(见 03_第3步结果.md)
PTB_KEY = "eval_dropless/paloma/ptb-llama3/loss"
CONTAM_STEPS = (206_999, 209_999, 212_999)  # #9874:这几次评测的 PTB 被污染,不参与拟合
PTB_RESIDUE = (-0.06, 0.0)  # 两次曝光可能给 PTB 留下的长期偏低(nats,PTB 自身);第二次约在第 36.2 万步
DD_SD_FLOOR = 0.02  # 阶梯上 ΔD 的标准差约 0.016;外推到 hero 规模,放宽到至少 0.02
TEAM = {"预注册 8/19": 2.039, "换配比修正 9/14": 2.02685, "重拟合 10/3(新配比)": 2.1001}


# ---------------- 数据 ----------------
def metric_steps(long, run, target):
    d = long[long.run == run]
    if target == "macro16":
        s = d[d.key == "eval_dropless/paloma/macro_loss"]
        return sorted(zip(s.step.astype(int), s.value.astype(float)))
    sub = d[d.key.str.match(r"^eval_dropless/paloma/[^/]+-llama3/loss$") & ~d.key.str.contains("/ptb-llama3/")]
    w = sub.pivot_table(index="step", columns="key", values="value").dropna()
    assert w.shape[1] == 15, (run, w.shape)
    return sorted(zip(w.index.astype(int), w.mean(axis=1).astype(float)))


def snap(series, stop, rule):
    grid = {}
    for step, v in series:
        x = step / stop if rule == "pre" else (step + 1) / stop
        grid[round(round(x * 20) / 20, 2)] = v
    return grid


def build_ladders(long, runs):
    lads = {
        "GB200": {"rungs": gb200_ladder(runs), "rule": "pre"},
        "H100-new": {"rungs": h100_ladder(runs, "new"), "rule": "refit"},
        "H100-old": {"rungs": h100_ladder(runs, "old"), "rule": "refit"},
    }
    for lad in lads.values():
        lad["grids"] = {t: {w: snap(metric_steps(long, v["run"], t), int(runs.loc[v["run"], "stop_after_steps"]), lad["rule"])
                            for w, v in lad["rungs"].items()} for t in TARGETS}
    return lads


def extend_d2048(grid, corr):
    g = dict(grid)
    win = [0.60, 0.65, 0.70, 0.75, 0.80]
    k, b = np.polyfit(win, [g[x] for x in win], 1)
    for p in FRACS:
        if p > 0.80:
            g[p] = k * p + b + corr
    return g


def hero_series(long, target):
    h = long[long.category == "hero"].drop_duplicates(["step", "key"])
    s = metric_steps(h.assign(run="hero"), "hero", target)
    s = [(st, v) for st, v in s if st > 0]
    p = np.array([(st + 1) / HERO_STEPS for st, _ in s])
    y = np.array([v for _, v in s])
    return p, y


# ---------------- 预测曲线与锚定 ----------------
def traj(grids, rungs, widths, conv, E, target_c):
    out = {}
    for p in FRACS:
        pts = [(rungs[w][conv] * p, grids[w][p]) for w in widths if p in grids[w]]
        if len(pts) < 2:
            continue
        c = np.array([x[0] for x in pts])
        y = np.array([x[1] for x in pts])
        s, i = np.polyfit(np.log(c), np.log(y - E), 1)
        out[p] = E + np.exp(i) * (target_c * p) ** s
    return out


def smooth(ys):
    ps = np.array([p for p in sorted(ys) if p >= SMOOTH_FROM - 1e-9])
    coef = np.polyfit(ps, np.array([ys[p] for p in ps]), SMOOTH_DEG)
    return lambda p: np.polyval(coef, p)


def anchor_fit(f, hp, hy, p_now, p_min=POST_SWAP_P):
    """残差 r(p) = y − f(p) 在 p ≥ p_min 上做线性拟合 r = a + b·(p − p_now)。返回 a、b 和它们的标准误。"""
    m = (hp >= p_min) & (hp <= p_now + 1e-9)
    x = hp[m] - p_now
    r = hy[m] - f(hp[m])
    X = np.column_stack([np.ones_like(x), x])
    beta = np.linalg.lstsq(X, r, rcond=None)[0]
    res = r - X @ beta
    s2 = res @ res / max(1, len(r) - 2)
    cov = s2 * np.linalg.inv(X.T @ X)
    return {"a": beta[0], "b": beta[1], "se_a": np.sqrt(cov[0, 0]), "se_b": np.sqrt(cov[1, 1]), "s": np.sqrt(s2), "n": len(r)}


def detrended_sd(p, y, p_min=POST_SWAP_P):
    m = p >= p_min
    x = np.log(p[m])
    c = np.polyfit(x, y[m], 2)
    return float(np.std(y[m] - np.polyval(c, x), ddof=3))


# ---------------- 伪 hero 回测 → 方法误差 ε ----------------
def pseudo_cases(lads):
    G, Hn, Ho = lads["GB200"], lads["H100-new"], lads["H100-old"]
    return [
        ("同阶梯 GB200: d1536 ← d768,d1024", G, ["d768", "d1024"], G, "d1536"),
        ("同阶梯 H100-new: d1536 ← d512–d1024", Hn, ["d512", "d768", "d1024"], Hn, "d1536"),
        ("同阶梯 H100-new: d1024 ← d512,d768", Hn, ["d512", "d768"], Hn, "d1024"),
        ("同阶梯 H100-old: d1024 ← d512,d768", Ho, ["d512", "d768"], Ho, "d1024"),
        ("跨阶梯: GB200 d1536 ← H100-new d512–d1024", Hn, ["d512", "d768", "d1024"], G, "d1536"),
        ("跨阶梯: H100-new d1536 ← GB200 d768,d1024", G, ["d768", "d1024"], Hn, "d1536"),
    ]


def e_grid(conv, step=0.05):
    lo, hi = E_PRIOR[conv]
    return np.round(np.arange(lo, hi + 1e-9, step), 3)


def backtest(lads, target):
    rows = []
    for name, train_lad, widths, tgt_lad, tgt_w in pseudo_cases(lads):
        tg = tgt_lad["grids"][target][tgt_w]
        if 1.0 not in tg:
            continue
        for p_now in (0.55, 0.60):
            hp = np.array([p for p in sorted(tg) if p <= p_now + 1e-9])
            hy = np.array([tg[p] for p in hp])
            for conv, cw in CONV_W.items():
                for E in e_grid(conv):
                    f = smooth(traj(train_lad["grids"][target], train_lad["rungs"], widths, conv, E,
                                    tgt_lad["rungs"][tgt_w][conv]))
                    an = anchor_fit(f, hp, hy, p_now)
                    rows.append({"case": name, "p_now": p_now, "conv": conv, "E": E,
                                 "pred": f(1.0) + an["a"], "actual": tg[1.0], "err": f(1.0) + an["a"] - tg[1.0],
                                 "a": an["a"], "b": an["b"], "se_a": an["se_a"], "se_b": an["se_b"], "s": an["s"],
                                 "f1": f(1.0)})
    if target == "macro16":  # 67B(#6044 的数字;锚点只有 21/63/80%,没法平滑,用单点锚)
        anchors = {
            512: {"tok80": 75.3e9, "C": {0.21: 5.28e18, 0.63: 1.58e19, 0.80: 2.00e19}, "L": {0.21: 3.4366, 0.63: 3.2925, 0.80: 3.2373}},
            768: {"tok80": 220.3e9, "C": {0.21: 3.44e19, 0.63: 1.04e20, 0.80: 1.31e20}, "L": {0.21: 3.1431, 0.63: 3.0048, 0.80: 2.9538}},
            1024: {"tok80": 542.4e9, "C": {0.21: 1.88e20, 0.63: 5.65e20, 0.80: 7.16e20}, "L": {0.21: 2.9241, 0.63: 2.8031, 0.80: 2.7515}},
        }
        tC = {0.21: 3.14e22, 0.63: 9.41e22, 0.80: 1.19e23}
        actual = {0.21: 2.398, 0.63: 2.300, 0.80: 2.256}

        def cc(d, C, p, conv):
            return C[p] if conv == "C_nolm" else C[p] + lm_flops(d, (C[p] / C[0.80]) * (8.02e12 if d == 2560 else anchors[d]["tok80"]))

        for p_now in (0.21, 0.63):
            for conv in CONV_W:
                for E in e_grid(conv):
                    pred = {}
                    for p in (p_now, 0.80):
                        c = np.array([cc(d, a["C"], p, conv) for d, a in anchors.items()])
                        y = np.array([a["L"][p] for a in anchors.values()])
                        s, i = np.polyfit(np.log(c), np.log(y - E), 1)
                        pred[p] = E + np.exp(i) * cc(2560, tC, p, conv) ** s
                    anchored = pred[0.80] + (actual[p_now] - pred[p_now])
                    rows.append({"case": "67B: 80% ← 3 个锚点", "p_now": p_now, "conv": conv, "E": E, "pred": anchored,
                                 "actual": actual[0.80], "err": anchored - actual[0.80], "a": actual[p_now] - pred[p_now],
                                 "b": 0.0, "se_a": 0.0, "se_b": 0.0, "s": 0.0, "f1": pred[0.80]})
    return pd.DataFrame(rows)


def case_errors(bt):
    """每个 (case, p_now) 先对 E 和口径取平均,得到一个误差。"""
    return bt.groupby(["case", "p_now"]).err.mean()


# ---------------- 蒙特卡洛 ----------------
def mc_predict(table_rows, p_now, sigma_eps, sigma_eval, n, p_out=None, kappa_fixed=None):
    """table_rows: list of dict(f1, a, b, se_a, se_b, traj{p: f(p)}, w);按权重抽样。返回终点样本(和轨迹样本)。
    kappa_fixed 只在稳健性检验里用:把漂移持续系数固定成一个值。"""
    w = np.array([r["w"] for r in table_rows], float)
    w /= w.sum()
    idx = RNG.choice(len(table_rows), size=n, p=w)
    f1 = np.array([table_rows[i]["f1"] for i in idx])
    a = np.array([table_rows[i]["a"] for i in idx]) + RNG.normal(size=n) * np.array([table_rows[i]["se_a"] for i in idx])
    b = np.array([table_rows[i]["b"] for i in idx]) + RNG.normal(size=n) * np.array([table_rows[i]["se_b"] for i in idx])
    kappa = RNG.uniform(0, 1, size=n) if kappa_fixed is None else np.full(n, float(kappa_fixed))
    eps = RNG.normal(size=n) * sigma_eps
    end = f1 + a + kappa * b * (1 - p_now) + eps + RNG.normal(size=n) * sigma_eval
    if p_out is None:
        return end, None
    tr = {}
    for p in p_out:
        fp = np.array([table_rows[i]["traj"][p] for i in idx])
        frac = max(0.0, (p - p_now) / (1 - p_now))
        tr[p] = fp + a + kappa * b * (p - p_now) + eps * frac + RNG.normal(size=n) * sigma_eval
    return end, tr


def hero_table(lads, target, hp, hy, p_now):
    rows = []
    for lad_name, lw in LADDER_W.items():
        lad = lads[lad_name]
        widths = list(lad["rungs"])
        corrs = np.round(np.arange(D2048_CORR[0], D2048_CORR[1] + 1e-9, 0.0005), 4) if lad_name == "GB200" else [None]
        for conv, cw in CONV_W.items():
            Es = e_grid(conv, step=0.01)
            for corr in corrs:
                grids = dict(lad["grids"][target])
                if corr is not None:
                    grids["d2048"] = extend_d2048(grids["d2048"], corr)
                for E in Es:
                    f = smooth(traj(grids, lad["rungs"], widths, conv, E, HERO[conv]))
                    an = anchor_fit(f, hp, hy, p_now)
                    rows.append({"ladder": lad_name, "conv": conv, "E": float(E), "corr": corr, "f1": float(f(1.0)),
                                 "a": an["a"], "b": an["b"], "se_a": an["se_a"], "se_b": an["se_b"], "s": an["s"],
                                 "n_anchor": an["n"], "traj": {p: float(f(p)) for p in P_OUT},
                                 "w": lw * cw / (len(Es) * len(corrs))})
    return rows


def quantiles(x):
    q = np.quantile(x, [0.025, 0.10, 0.50, 0.90, 0.975])
    return {"q025": q[0], "q10": q[1], "median": q[2], "q90": q[3], "q975": q[4], "mean": float(np.mean(x)), "sd": float(np.std(x))}


def coverage_check(lads, target, bt):
    """留一个案例:用其他案例估 σ_ε,对这个案例按同一套方法给区间,看实际值落不落在里面。"""
    ce = case_errors(bt)
    rows = []
    for (case, p_now), _ in ce.items():
        others = ce.drop(index=(case, p_now))
        others = others[[c != case for c, _ in others.index]]  # 同一案例的另一个锚点也排除
        sig = float(np.sqrt(np.mean(others.values ** 2)))
        sub = bt[(bt.case == case) & (bt.p_now == p_now)]
        tab = [{"f1": r.f1, "a": r.a, "b": r.b, "se_a": r.se_a, "se_b": r.se_b, "w": CONV_W[r.conv]} for r in sub.itertuples()]
        s_eval = float(sub.s.mean())
        end, _ = mc_predict(tab, p_now if not case.startswith("67B") else p_now, sig, s_eval, 20_000)
        if case.startswith("67B"):  # 67B 的终点是 80%,漂移项按 0.80 计
            end = np.array([r["f1"] + r["a"] for r in tab])[RNG.integers(0, len(tab), 20_000)] + RNG.normal(size=20_000) * sig
        q = quantiles(end)
        act = float(sub.actual.iloc[0])
        rows.append({"case": case, "p_now": p_now, "sigma_eps_loo": sig, "median": q["median"], "actual": act,
                     "in80": q["q10"] <= act <= q["q90"], "in95": q["q025"] <= act <= q["q975"]})
    return pd.DataFrame(rows)


def decompose(rows, p_now, sigma_eps, sigma_eval):
    """按阶梯分别给出:不漂移时的中心值、阶梯内 θ(口径、E、d2048 修正)造成的标准差、κ=1 时漂移的大小;
    再加上和阶梯无关的几项。"""
    out = {}
    for lad in LADDER_W:
        sub = [r for r in rows if r["ladder"] == lad]
        w = np.array([r["w"] for r in sub])
        x = np.array([r["f1"] + r["a"] for r in sub])
        drift1 = np.array([r["b"] * (1 - p_now) for r in sub])
        out[lad] = {"不漂移时的中心值": float(np.average(x, weights=w)),
                    "阶梯内 θ 的标准差": float(np.sqrt(np.average((x - np.average(x, weights=w)) ** 2, weights=w))),
                    "κ=1 时的漂移": float(np.average(drift1, weights=w)),
                    "漂移斜率的标准误 × 剩余进度": float(np.average([r["se_b"] for r in sub], weights=w) * (1 - p_now))}
    out["锚定偏移的标准误"] = float(np.average([r["se_a"] for r in rows], weights=[r["w"] for r in rows]))
    out["方法误差 ε(回测)"] = float(sigma_eps)
    out["终点评测噪声"] = float(sigma_eval)
    return out


# ---------------- PTB 这一项:D = PTB − 15 子集 macro ----------------
def key_steps(long, run, key):
    d = long[(long.run == run) & (long.key == key)]
    return sorted(zip(d.step.astype(int), d.value.astype(float)))


def ptb_gap_model(long, runs, lads, p_now):
    deltas = []
    for name in ("GB200", "H100-new", "H100-old"):
        lad = lads[name]
        for w, v in lad["rungs"].items():
            g15 = lad["grids"]["macro15"][w]
            gp = snap(key_steps(long, v["run"], PTB_KEY), int(runs.loc[v["run"], "stop_after_steps"]), lad["rule"])
            if all(x in g15 and x in gp for x in (0.55, 0.60, 1.0)):
                d = {x: gp[x] - g15[x] for x in (0.55, 0.60, 1.0)}
                deltas.append({"ladder": name, "width": w, "dD": d[1.0] - np.interp(p_now, [0.55, 0.60], [d[0.55], d[0.60]])})
    h = long[long.category == "hero"].drop_duplicates(["step", "key"])
    w = h.pivot_table(index="step", columns="key", values="value").sort_index()
    sub = [c for c in w.columns if c.startswith("eval_dropless/paloma/") and c.endswith("-llama3/loss")]
    D = (w[PTB_KEY] - w[[c for c in sub if "/ptb-" not in c]].mean(axis=1)).dropna()
    D = D[(D.index >= 116_999) & ~D.index.isin(CONTAM_STEPS)]
    p = (D.index.values + 1) / HERO_STEPS
    X = np.column_stack([np.ones_like(p), p - p_now])
    beta = np.linalg.lstsq(X, D.values, rcond=None)[0]
    res = D.values - X @ beta
    s2 = res @ res / (len(res) - 2)
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[0, 0])
    dd = np.array([x["dD"] for x in deltas])
    return {"D_now": float(beta[0]), "se_D_now": float(se), "hero_slope": float(beta[1]), "sd_noise": float(np.sqrt(s2)),
            "n_hero": int(len(res)), "dD_mean": float(dd.mean()), "dD_sd": float(dd.std(ddof=1)),
            "dD_sd_used": float(max(dd.std(ddof=1), DD_SD_FLOOR)), "rungs": deltas,
            "hero_dD_if_linear": float(beta[1] * (1 - p_now))}


def sample_D(pt, n, p_now, p_out):
    d_now = pt["D_now"] + RNG.normal(size=n) * pt["se_D_now"]
    dd = RNG.normal(pt["dD_mean"], pt["dD_sd_used"], size=n)
    residue = RNG.uniform(*PTB_RESIDUE, size=n)
    end = d_now + dd + residue + RNG.normal(size=n) * pt["sd_noise"]
    traj = {}
    for p in p_out:
        frac = max(0.0, (p - p_now) / (1 - p_now))
        traj[p] = d_now + dd * frac + residue * frac + RNG.normal(size=n) * pt["sd_noise"]
    return end, traj


def plot(results, hero, path):
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=False)
    for ax, target in zip(axes, ("macro16", "macro15")):
        hp, hy = hero[target]
        ax.plot(hp * 100, hy, "-", color="black", lw=1.4, label="hero 实测")
        tr = results[target]["trajectory"]
        ps = np.array([float(p) for p in tr]) * 100
        med = np.array([tr[p]["median"] for p in tr])
        ax.fill_between(ps, [tr[p]["q025"] for p in tr], [tr[p]["q975"] for p in tr], color="#1f77b4", alpha=0.18, label="95% 区间")
        ax.fill_between(ps, [tr[p]["q10"] for p in tr], [tr[p]["q90"] for p in tr], color="#1f77b4", alpha=0.35, label="80% 区间")
        ax.plot(ps, med, "-", color="#1f77b4", lw=2, label="中位数")
        if target == "macro16":
            dtr = results["macro16_direct"]["trajectory"]
            ax.plot(ps, [dtr[p]["median"] for p in dtr], "--", color="gray", lw=1.2, label="对照:直接锚定官方 macro 的中位数")
            for (name, v), col in zip(TEAM.items(), ("crimson", "darkorange", "seagreen")):
                ax.plot(100, v, "D", color=col, ms=8, label=f"团队{name}:{v:.3f}")
        ax.set_xlim(25, 102)
        lo = min(hy[hp > 0.25].min(), min(tr[p]["q025"] for p in tr)) - 0.02
        ax.set_ylim(lo, hy[hp > 0.25].max() + 0.01)
        ax.set_title("官方 16 子集 macro(主目标)" if target == "macro16" else "去掉 PTB 的 15 子集 macro(伴随目标)")
        ax.set_xlabel("训练进度(%)")
        ax.set_ylabel("dropless Paloma loss")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8.5, loc="upper right")
    fig.suptitle(f"锚定法预测(数据截至第 {results['macro16']['data_until_step']:,} 步)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def summarize(end, tr, extra):
    q = quantiles(end)
    out = {"end": q,
           "probs": {"P(≤2.039)": float(np.mean(end <= 2.039)), "P(≤2.027)": float(np.mean(end <= 2.02685)),
                     "P(≥2.100)": float(np.mean(end >= 2.100)), "P(2.02–2.06)": float(np.mean((end >= 2.02) & (end <= 2.06)))},
           "team_percentile": {k: float(np.mean(end <= v)) for k, v in TEAM.items()}}
    if tr is not None:
        out["trajectory"] = {f"{p:.2f}": quantiles(tr[p]) for p in P_OUT}
    out.update(extra)
    return out


def main(snap_dir, outdir):
    os.makedirs(outdir, exist_ok=True)
    long, runs = load(snap_dir)
    lads = build_ladders(long, runs)
    results, hero, samples, all_bt, all_cov = {}, {}, {}, [], []
    for target in TARGETS:
        hp, hy = hero_series(long, target)
        hero[target] = (hp, hy)
        p_now = float(hp[-1])
        sigma_eval = detrended_sd(hp, hy)

        bt = backtest(lads, target)
        bt.insert(0, "target", target)
        ce = case_errors(bt)
        sigma_eps = float(np.sqrt(np.mean(ce.values ** 2)))
        cov = coverage_check(lads, target, bt)
        cov.insert(0, "target", target)
        all_bt.append(bt)
        all_cov.append(cov)

        rows = hero_table(lads, target, hp, hy, p_now)
        end, tr = mc_predict(rows, p_now, sigma_eps, sigma_eval, N_MC, P_OUT)
        samples[target] = (end, tr)
        by_ladder = {}
        for lad_name in LADDER_W:
            sub = [r for r in rows if r["ladder"] == lad_name]
            e_, _ = mc_predict(sub, p_now, sigma_eps, sigma_eval, 30_000)
            by_ladder[lad_name] = quantiles(e_)
        key = "macro15" if target == "macro15" else "macro16_direct"
        results[key] = summarize(end, tr, {
            "data_until_step": int(round(p_now * HERO_STEPS - 1)), "p_now": p_now, "hero_last": float(hy[-1]),
            "sigma_eval": sigma_eval, "sigma_eps": sigma_eps,
            "case_errors": {f"{c} @{p}": float(v) for (c, p), v in ce.items()}, "by_ladder": by_ladder,
            "anchor_center": {k: float(np.average([r[k] for r in rows], weights=[r["w"] for r in rows])) for k in ("f1", "a", "b", "se_a", "se_b")},
            "decomposition": decompose(rows, p_now, sigma_eps, sigma_eval)})

    # 官方 16 子集 macro(主目标)= 15 子集 macro + D/16
    p_now = results["macro15"]["p_now"]
    pt = ptb_gap_model(long, runs, lads, p_now)
    d_end, d_tr = sample_D(pt, N_MC, p_now, P_OUT)
    end15, tr15 = samples["macro15"]
    end16 = end15 + d_end / 16
    tr16 = {p: tr15[p] + d_tr[p] / 16 for p in P_OUT}
    results["macro16"] = summarize(end16, tr16, {
        "method": "15 子集 macro 的锚定预测 + (PTB − 15 子集)/16", "data_until_step": results["macro15"]["data_until_step"],
        "p_now": p_now, "hero_last": float(hero["macro16"][1][-1]), "ptb_model": pt,
        "D_end": quantiles(d_end), "D_contribution_over_16": quantiles(d_end / 16)})

    pd.concat(all_bt).to_csv(os.path.join(outdir, "backtest.csv"), index=False)
    pd.concat(all_cov).to_csv(os.path.join(outdir, "coverage.csv"), index=False)
    json.dump(results, open(os.path.join(outdir, "forecast_step3.json"), "w"), ensure_ascii=False, indent=1)
    # 终点的蒙特卡洛样本:训练结束后用来精确计算 CRPS
    np.savez_compressed(os.path.join(outdir, "end_samples.npz"), macro16=end16.astype(np.float32),
                        macro15=end15.astype(np.float32))
    plot(results, hero, os.path.join(outdir, "forecast_step3.png"))

    for key in ("macro16", "macro15", "macro16_direct"):
        r = results[key]
        e = r["end"]
        print(f"\n===== {key} =====  数据截至第 {r['data_until_step']} 步(p={r['p_now']:.4f}),最后一个实测 {r['hero_last']:.4f}")
        print(f"终点:中位数 {e['median']:.4f};80% 区间 [{e['q10']:.4f}, {e['q90']:.4f}];95% 区间 [{e['q025']:.4f}, {e['q975']:.4f}];SD {e['sd']:.4f}")
        print("概率:", {k: round(v, 3) for k, v in r["probs"].items()})
        print("团队点预测在我们分布里的分位:", {k: round(v, 3) for k, v in r["team_percentile"].items()})
        if "by_ladder" in r:
            print("按阶梯:", {k: f"{v['median']:.4f} [{v['q10']:.4f}, {v['q90']:.4f}]" for k, v in r["by_ladder"].items()})
            print(f"σ_eval={r['sigma_eval']:.4f}  σ_ε={r['sigma_eps']:.4f}")
            print("分解:", json.dumps(r["decomposition"], ensure_ascii=False, default=lambda v: round(v, 4)))
        if "ptb_model" in r:
            pm = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r["ptb_model"].items() if k != "rungs"}
            print("PTB 模型:", pm)
            print("D_end:", {k: round(v, 4) for k, v in r["D_end"].items()}, " D_end/16:", {k: round(v, 4) for k, v in r["D_contribution_over_16"].items()})
        if "trajectory" in r:
            print("轨迹:", {p: f"{v['median']:.4f} [{v['q10']:.4f}, {v['q90']:.4f}]" for p, v in r["trajectory"].items()})
    cov = pd.concat(all_cov)
    print("\n===== 留一校准检查 =====")
    print(cov.groupby("target")[["in80", "in95"]].mean())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
