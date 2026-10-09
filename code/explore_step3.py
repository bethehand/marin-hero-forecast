#!/usr/bin/env python3
"""第 3 步设计前的探索实验:用已知答案检验各种做法准不准。

E1 留出最大一级:在一组阶梯里用小的几级拟合,预测最大一级(真实值已知)。
   比较 C 的两种口径(不含 / 含 lm_head)× 两种拟合法(E 固定、逐进度 log 拟合 / E 拟合、各进度共用、loss 空间)。
E2 67B 回测(#6044):3 个锚点 → 67B 在 21% / 63% / 80% 处的实际值,同样比较口径和拟合法。
E3 单条曲线外推:每个跑完的 rung 只用前一段(到 55% 或 60%)的评测点,按学习率调度拟合
   L(p) = L0 + A·S1(p)^(−α) − B·S2(p)(Tissue 等 2024 的形式;S1 = 学习率累积面积,S2 = 退火面积),
   预测它自己 100% 处的 loss,看误差。

用法:
  python explore_step3.py ../data/snapshot_2026-10-09 ../results/step3_explore
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize_scalar

from reproduce import PRE_FLOPS, fit_group, load, prereg_grid, refit_grid, total_sse

V = 128_256
SEQ = 4096
FRACS_30_100 = [round(0.30 + 0.05 * i, 2) for i in range(15)]


def lm_flops(d, tokens):
    return 6.0 * d * V * tokens


# ---------- 阶梯定义:宽度、token 数、两种口径的 C ----------
def gb200_ladder(runs):
    out = {}
    for w, run in (("d768", "rav-ladder-d768-v2"), ("d1024", "rav-ladder-d1024"), ("d1536", "rav-ladder-d1536"),
                   ("d2048", "rav-ladder-d2048-v3")):
        c_per_step, steps = PRE_FLOPS[w]
        batch = {"d768": 1024, "d1024": 2048, "d1536": 6144, "d2048": 11264}[w]
        tokens = steps * batch * SEQ
        nolm = c_per_step * steps
        out[w] = {"run": run, "d": int(w[1:]), "tokens": tokens, "C_nolm": nolm, "C_lm": nolm + lm_flops(int(w[1:]), tokens)}
    return out


def h100_ladder(runs, mix):
    names = {
        "new": {"d512": "h100-mix25-20260912-d512", "d768": "h100-mix25-20260912-d768",
                "d1024": "h100-mix25-20260912-d1024", "d1536": "h100-mix25-20260912-d1536"},
        "old": {"d512": "h100-ladder-d512-ep8-bs1024-791tpp-20260824-rno2a",
                "d768": "h100-ladder-d768-2xep8-bs1024-791tpp-20260824",
                "d1024": "h100-ladder-d1024-ep8x4-bs1024-791tpp-20260825-rno2a-interactive",
                "d1536": "h100-ladder-d1536-ep8x8-bs1024-791tpp-20260825-rno2a-interactive"},
    }[mix]
    c_lm = {"d512": 1.0016130022244352e19, "d768": 5.480525398095692e19, "d1024": 2.735902188489325e20,
            "d1536": 1.831305999780126e21}  # 10/3 重拟合 config 的 source_compute(含 lm_head)
    out = {}
    for w, run in names.items():
        tokens = int(runs.loc[run, "stop_after_steps"]) * 1024 * SEQ
        out[w] = {"run": run, "d": int(w[1:]), "tokens": tokens, "C_lm": c_lm[w], "C_nolm": c_lm[w] - lm_flops(int(w[1:]), tokens)}
    return out


HERO = {"C_nolm": PRE_FLOPS["d6144"][0] * PRE_FLOPS["d6144"][1]}
HERO["C_lm"] = HERO["C_nolm"] + lm_flops(6144, 390_251 * 11_264 * SEQ)


# ---------- 两种拟合法 ----------
def fit_fixedE(E, pts):
    """逐进度 log 空间:log(L − E) = log A − α log C。pts = [(C, L)]。"""
    c = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    s, i = np.polyfit(np.log(c), np.log(y - E), 1)
    return np.exp(i), -s


def fit_sharedF(groups):
    """loss 空间,所有组共用一个下界 F;groups = {key: [(width, C, L)]}。"""
    lmin = min(p[2] for v in groups.values() for p in v)
    r = minimize_scalar(lambda F: total_sse(F, groups), bounds=(0.0, lmin - 1e-4), method="bounded",
                        options={"xatol": 1e-7})
    return r.x


def predict(method, train, target_c, conv):
    """train = {fraction: [(width, C_conv, L)]};返回 {fraction: 预测值}。"""
    preds = {}
    if method.startswith("E="):
        E = float(method[2:])
        for p, pts in train.items():
            A, a = fit_fixedE(E, [(c, L) for (_, c, L) in pts])
            preds[p] = E + A * target_c[p] ** (-a)
    else:
        F = fit_sharedF({("g", p): v for p, v in train.items()})
        for p, pts in train.items():
            A, a, _ = fit_group(F, pts)
            preds[p] = F + A * target_c[p] ** (-a)
        preds["_F"] = F
    return preds


METHODS = ["E=1.3", "E=1.5", "E=1.7", "sharedF"]


def e1_leave_largest_out(long, runs):
    rows = []
    cases = []
    gb = gb200_ladder(runs)
    grids = {w: prereg_grid(long, v["run"], int(runs.loc[v["run"], "stop_after_steps"])) for w, v in gb.items()}
    cases.append(("GB200:d768–d1536→d2048", gb, grids, ["d768", "d1024", "d1536"], "d2048",
                  [p for p in FRACS_30_100 if p <= 0.80]))
    for mix in ("new", "old"):
        h = h100_ladder(runs, mix)
        g = {w: refit_grid(long, v["run"], int(runs.loc[v["run"], "stop_after_steps"])) for w, v in h.items()}
        fr = [p for p in FRACS_30_100 if p in g["d1536"]]
        cases.append((f"H100-{mix}:d512–d1024→d1536", h, g, ["d512", "d768", "d1024"], "d1536", fr))
    for name, lad, grids, train_w, target_w, fracs in cases:
        for conv in ("C_nolm", "C_lm"):
            train = {p: [(w, lad[w][conv] * p, grids[w][p]) for w in train_w if p in grids[w]] for p in fracs}
            target_c = {p: lad[target_w][conv] * p for p in fracs}
            for m in METHODS:
                pr = predict(m, train, target_c, conv)
                err = np.array([pr[p] - grids[target_w][p] for p in fracs])
                rows.append({"case": name, "conv": conv, "method": m, "n_fracs": len(fracs),
                             "mean_err": err.mean(), "rmse": np.sqrt((err ** 2).mean()),
                             "err_last": err[-1], "F": pr.get("_F")})
    return pd.DataFrame(rows)


def e2_67b():
    # #6044 的表:no-lm 的 C(SWA-aware)和 loss;锚点在 21% / 63% / 80% 三个进度
    anchors = {
        "d512": {"d": 512, "tok80": 75.3e9, "C": {0.21: 5.28e18, 0.63: 1.58e19, 0.80: 2.00e19},
                 "L": {0.21: 3.4366, 0.63: 3.2925, 0.80: 3.2373}},
        "d768": {"d": 768, "tok80": 220.3e9, "C": {0.21: 3.44e19, 0.63: 1.04e20, 0.80: 1.31e20},
                 "L": {0.21: 3.1431, 0.63: 3.0048, 0.80: 2.9538}},
        "d1024": {"d": 1024, "tok80": 542.4e9, "C": {0.21: 1.88e20, 0.63: 5.65e20, 0.80: 7.16e20},
                  "L": {0.21: 2.9241, 0.63: 2.8031, 0.80: 2.7515}},
    }
    target = {"d": 2560, "C": {0.21: 3.14e22, 0.63: 9.41e22, 0.80: 1.19e23}, "tok80": 8.02e12,
              "actual": {0.21: 2.398, 0.63: 2.300, 0.80: 2.256}}

    def conv_c(entry, p, conv):
        c = entry["C"][p]
        if conv == "C_nolm":
            return c
        tokens = entry["tok80"] * (c / entry["C"][0.80])  # token 数和 no-lm 的 C 成正比
        return c + lm_flops(entry["d"], tokens)

    rows = []
    fracs = [0.21, 0.63, 0.80]
    for conv in ("C_nolm", "C_lm"):
        train = {p: [(w, conv_c(a, p, conv), a["L"][p]) for w, a in anchors.items()] for p in fracs}
        tc = {p: conv_c(target, p, conv) for p in fracs}
        for m in ["E=1.2", "E=1.3", "E=1.4", "E=1.5", "sharedF"]:
            pr = predict(m, train, tc, conv)
            for p in fracs:
                rows.append({"conv": conv, "method": m, "fraction": p, "pred": pr[p], "actual": target["actual"][p],
                             "err": pr[p] - target["actual"][p], "F": pr.get("_F")})
    return pd.DataFrame(rows)


# ---------- E3:单条曲线按学习率调度外推 ----------
def lr_schedule(T, warm=0.01, min_ratio=0.05):
    t = np.arange(T, dtype=float)
    tw = max(1, int(round(warm * T)))
    eta = np.where(t < tw, (t + 1) / tw, 1.0 - (1.0 - min_ratio) * (t - tw) / max(1, (T - tw)))
    return eta


def areas(T, lam=0.999):
    eta = lr_schedule(T)
    s1 = np.cumsum(eta)
    d = np.concatenate([[0.0], -np.diff(eta)])
    d[d < 0] = 0.0  # 预热阶段学习率上升,不算退火
    m = np.zeros(T)
    for i in range(1, T):
        m[i] = lam * m[i - 1] + d[i]
    s2 = np.cumsum(m)
    return s1 / T, s2 / T  # 归一化到"每单位训练进度"


def fit_curve(p, y, s1, s2):
    """L = L0 + A·S1^(−α) − B·S2;α∈(0,3),A>0,B≥0。"""
    def resid(x):
        L0, lnA, al, B = x
        return L0 + np.exp(lnA) * s1 ** (-al) - B * s2 - y

    best = None
    for al0 in (0.3, 0.6, 1.0):
        x0 = [y.min() - 0.3, np.log(0.3), al0, 0.0]
        sol = least_squares(resid, x0, bounds=([0.0, -20, 1e-3, 0.0], [y.min(), 20, 3.0, 50.0]), xtol=1e-12, ftol=1e-12)
        if best is None or sol.cost < best.cost:
            best = sol
    return best.x


def e3_single_curve(long, runs):
    sets = []
    gb = gb200_ladder(runs)
    for w in ("d768", "d1024", "d1536"):
        sets.append(("GB200", w, gb[w]["run"], "pre"))
    for mix in ("new", "old"):
        for w, v in h100_ladder(runs, mix).items():
            sets.append((f"H100-{mix}", w, v["run"], "refit"))
    rows = []
    for lad, w, run, rule in sets:
        T = int(runs.loc[run, "stop_after_steps"])
        grid = (prereg_grid if rule == "pre" else refit_grid)(long, run, T)
        if 1.0 not in grid:
            continue
        s1, s2 = areas(T)
        for cut in (0.55, 0.60, 0.80):
            ps = np.array(sorted(p for p in grid if p <= cut + 1e-9 and p >= 0.10))
            idx = np.clip((ps * T).astype(int) - 1, 0, T - 1)
            y = np.array([grid[p] for p in ps])
            L0, lnA, al, B = fit_curve(ps, y, s1[idx], s2[idx])
            pred = L0 + np.exp(lnA) * s1[T - 1] ** (-al) - B * s2[T - 1]
            # 对照:团队做 d2048 用的 60–80% 线性外推(只有 cut=0.80 时能做)
            lin = None
            if cut >= 0.80:
                win = [0.60, 0.65, 0.70, 0.75, 0.80]
                k, b = np.polyfit(win, [grid[x] for x in win], 1)
                lin = k + b
            rows.append({"ladder": lad, "width": w, "cut": cut, "n_points": len(ps), "pred_100": pred,
                         "actual_100": grid[1.0], "err": pred - grid[1.0],
                         "lin6080_err": (lin - grid[1.0]) if lin is not None else None})
    return pd.DataFrame(rows)


# ---------- E4:锚定法的回测(拿已知答案的"伪 hero"检验)----------
def per_fraction_traj(train, target_c, E):
    """train = {p: [(C, L)]} → {p: 预测值},逐进度 log 空间、E 固定。"""
    out = {}
    for p, pts in train.items():
        if len(pts) >= 2:
            A, a = fit_fixedE(E, pts)
            out[p] = E + A * target_c[p] ** (-a)
    return out


def e4_anchored_backtest(long, runs):
    rows = []
    fr = [round(0.05 * i, 2) for i in range(1, 21)]
    cases = []
    gb = gb200_ladder(runs)
    g_gb = {w: prereg_grid(long, v["run"], int(runs.loc[v["run"], "stop_after_steps"])) for w, v in gb.items()}
    cases.append(("GB200 d1536 ← d768,d1024", gb, g_gb, ["d768", "d1024"], "d1536"))
    for mix, target, train_w in (("new", "d1536", ["d512", "d768", "d1024"]), ("old", "d1024", ["d512", "d768"])):
        h = h100_ladder(runs, mix)
        g = {w: refit_grid(long, v["run"], int(runs.loc[v["run"], "stop_after_steps"])) for w, v in h.items()}
        cases.append((f"H100-{mix} {target} ← {','.join(train_w)}", h, g, train_w, target))
    for name, lad, grids, train_w, target_w in cases:
        for conv in ("C_nolm", "C_lm"):
            train = {p: [(lad[w][conv] * p, grids[w][p]) for w in train_w if p in grids[w]] for p in fr}
            tc = {p: lad[target_w][conv] * p for p in fr}
            actual_end = grids[target_w][1.0]
            for anchor_p in (0.30, 0.55, 0.60):
                if anchor_p not in grids[target_w]:
                    continue
                for E in (1.3, 1.5, 1.7, 1.9):
                    t = per_fraction_traj(train, tc, E)
                    absolute = t[1.0]
                    anchored = grids[target_w][anchor_p] - (t[anchor_p] - t[1.0])
                    rows.append({"case": name, "conv": conv, "anchor_p": anchor_p, "E": E,
                                 "abs_err": absolute - actual_end, "anchored_err": anchored - actual_end})
    # 67B:锚点只有 21% / 63% / 80%;在 21% 或 63% 锚定,预测 80%
    e2 = e2_67b()  # 只为拿到同一套常数;下面重算
    anchors = {
        "d512": {"d": 512, "tok80": 75.3e9, "C": {0.21: 5.28e18, 0.63: 1.58e19, 0.80: 2.00e19}, "L": {0.21: 3.4366, 0.63: 3.2925, 0.80: 3.2373}},
        "d768": {"d": 768, "tok80": 220.3e9, "C": {0.21: 3.44e19, 0.63: 1.04e20, 0.80: 1.31e20}, "L": {0.21: 3.1431, 0.63: 3.0048, 0.80: 2.9538}},
        "d1024": {"d": 1024, "tok80": 542.4e9, "C": {0.21: 1.88e20, 0.63: 5.65e20, 0.80: 7.16e20}, "L": {0.21: 2.9241, 0.63: 2.8031, 0.80: 2.7515}},
    }
    target = {"d": 2560, "tok80": 8.02e12, "C": {0.21: 3.14e22, 0.63: 9.41e22, 0.80: 1.19e23}, "actual": {0.21: 2.398, 0.63: 2.300, 0.80: 2.256}}

    def cc(e, p, conv):
        c = e["C"][p]
        return c if conv == "C_nolm" else c + lm_flops(e["d"], e["tok80"] * c / e["C"][0.80])

    for conv in ("C_nolm", "C_lm"):
        train = {p: [(cc(a, p, conv), a["L"][p]) for a in anchors.values()] for p in (0.21, 0.63, 0.80)}
        tc = {p: cc(target, p, conv) for p in (0.21, 0.63, 0.80)}
        for anchor_p in (0.21, 0.63):
            for E in (1.2, 1.3, 1.4, 1.5, 1.6, 1.8):
                t = per_fraction_traj(train, tc, E)
                anchored = target["actual"][anchor_p] - (t[anchor_p] - t[0.80])
                rows.append({"case": "67B 80% ← 3 锚点", "conv": conv, "anchor_p": anchor_p, "E": E,
                             "abs_err": t[0.80] - target["actual"][0.80], "anchored_err": anchored - target["actual"][0.80]})
    return pd.DataFrame(rows)


def main(snap, outdir):
    os.makedirs(outdir, exist_ok=True)
    long, runs = load(snap)
    pd.set_option("display.width", 200)

    e1 = e1_leave_largest_out(long, runs)
    e1.to_csv(os.path.join(outdir, "e1_leave_largest_out.csv"), index=False)
    print("== E1 留出最大一级(误差 = 预测 − 实际)==")
    print(e1.round(4).to_string(index=False))

    e2 = e2_67b()
    e2.to_csv(os.path.join(outdir, "e2_67b_backtest.csv"), index=False)
    print("\n== E2 67B 回测 ==")
    print(e2.round(4).to_string(index=False))

    e3 = e3_single_curve(long, runs)
    e3.to_csv(os.path.join(outdir, "e3_single_curve.csv"), index=False)
    print("\n== E3 单条曲线外推到 100% ==")
    print(e3.round(4).to_string(index=False))
    print("\n按截断点汇总:")
    print(e3.groupby("cut").err.agg(["count", "mean", "std", lambda s: np.sqrt((s ** 2).mean())]).round(4))

    e4 = e4_anchored_backtest(long, runs)
    e4.to_csv(os.path.join(outdir, "e4_anchored_backtest.csv"), index=False)
    print("\n== E4 锚定法回测(误差 = 预测 − 实际)==")
    print(e4.round(4).to_string(index=False))
    s = e4.groupby(["case", "anchor_p"]).agg(abs_err_range=("abs_err", lambda x: x.max() - x.min()),
                                             anchored_err_mean=("anchored_err", "mean"),
                                             anchored_err_range=("anchored_err", lambda x: x.max() - x.min()))
    print("\n汇总(跨 E 和口径):")
    print(s.round(4).to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
